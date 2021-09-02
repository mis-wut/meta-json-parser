#pragma once
#include <cuda_runtime_api.h>
#include <utility>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/static_buffer.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/output_printer.cuh>
#include <meta_json_parser/parser_requirements.cuh>
#include <meta_json_parser/intelisense_silencer.h>
#include <cub/cub.cuh>

//Only parsing/validation
struct JString
{
	using MemoryRequests = JsonParse::StringRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		return JsonParse::String(kc, [](bool&, int&){ return ParsingError::None; });
	}
};

template<class BytesT, class TagT>
struct JStringStaticCopy
{
	static_assert(BytesT::value > 0, "BytesT must be at greater than 0");

	using type = JStringStaticCopy<BytesT, TagT>;
	using Tag = TagT;
	using Printer = AsCharsPrinter<type>;
	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, StaticBuffer_c<BytesT::value>>>;
	using MemoryRequests = JsonParse::StringRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		using RT = typename KC::RT;
		char (&result)[BytesT::value] = kc.om.template Get<KernelContextT, TagT>().template Alias<char[BytesT::value]>();
		uint32_t offset = 0;
		ParsingError err = JsonParse::String(kc, [&](bool& isEscaped, int& activeThreads) {
			uint32_t worker_offset = offset + RT::WorkerId();
			char c = RT::WorkerId() < activeThreads ? kc.wgr.CurrentChar() : '\0';
			if (worker_offset < BytesT::value)
				result[worker_offset] = c;
			offset += RT::GroupSize();
			return ParsingError::None;
		});
		if (err != ParsingError::None)
			return err;
		while (offset < BytesT::value)
		{
			uint32_t worker_offset = offset + RT::WorkerId();
			if (worker_offset < BytesT::value)
				result[worker_offset] = '\0';
			offset += RT::GroupSize();
		}
		return ParsingError::None;
	}
};

struct IsNotNullByte {
	__device__ bool operator()(char& val)
	{
		return val != '\0';
	}
};


template<class ActionT>
struct DynamicStringPrinter
{
	using Tag = typename ActionT::Tag;
	using LengthRequestTag = typename ActionT::LengthRequestTag;
	using DynamicStringRequestTag = typename ActionT::DynamicStringRequestTag;

	template<class ParserOutputHostT>
	void static Print(const ParserOutputHostT& output, size_t idx, std::ostream& stream)
	{
		const uint32_t begin = reinterpret_cast<const uint32_t*>(output.template Pointer<LengthRequestTag>())[idx];
		const uint32_t end   = reinterpret_cast<const uint32_t*>(output.template Pointer<LengthRequestTag>())[idx + 1];
		const char* ptr = reinterpret_cast<const char*>(output.template Pointer<DynamicStringRequestTag>());
		stream.write(ptr + begin, end - begin);
	}
};

template<class TagT>
struct JStringDynamicCopy
{
	using type = JStringDynamicCopy<TagT>;
	using Tag = TagT;
	using LengthRequestTag = std::pair<TagT, boost::mp11::mp_int<0>>;
	using LengthRequest = OutputRequest<
		LengthRequestTag,
		uint32_t,
		boost::mp11::mp_list<
			OutputOptElementsBefore_c<1>
		>
	>;
	using DynamicStringRequestTag = std::pair<TagT, boost::mp11::mp_int<1>>;
	using DynamicStringRequest = DynamicOutputRequest<DynamicStringRequestTag, char>;
	using Printer = DynamicStringPrinter<type>;
	using OutputRequests = boost::mp11::mp_list<LengthRequest, DynamicStringRequest>;
	using MemoryRequests = JsonParse::StringRequests;

	//excpect ParserKernel
	template<class PK>
	static void PostKernelHook(
		PK& pk,
		const char* input,
		const InputIndex* indices,
		ParsingError* errors,
		const uint32_t count,
		void** h_outputs
	)
	{
		using OM = OutputManager<typename PK::BaseAction>;

		uint32_t* lengths = reinterpret_cast<uint32_t*>(
			h_outputs[OM::template TagIndex<LengthRequestTag>::value]
		);
		char* content = reinterpret_cast<char*>(
			h_outputs[OM::template TagIndex<DynamicStringRequestTag>::value]
		);

		auto dynamic_size = pk.m_launch_config->dynamic_sizes[OM::template DynamicTagIndex<DynamicStringRequestTag>::value];

		const size_t CUB_BUFFER_SIZE = PK::CUB_BUFFER_SIZE;

		size_t* d_num_selected_out = reinterpret_cast<size_t*>(pk.m_cub_buffer);
		uint8_t* d_temp_storage = pk.m_cub_buffer + 256;
		size_t temp_storage_bytes = 0;

		cub::DeviceSelect::If(
			nullptr,
			temp_storage_bytes,
			content,
			content,
			d_num_selected_out,
			count * dynamic_size,
			IsNotNullByte(),
			pk.m_stream
		);

		if (temp_storage_bytes > CUB_BUFFER_SIZE - 256)
		{
			std::cerr << "Fatal. Not enough CUB_BUFFER. " << temp_storage_bytes + 256 << " < " << CUB_BUFFER_SIZE << "\n";
			exit(1);
		}

		cub::DeviceSelect::If(
			d_temp_storage,
			temp_storage_bytes,
			content,
			content,
			d_num_selected_out,
			count * dynamic_size,
			IsNotNullByte(),
			pk.m_stream
		);

		cub::DeviceScan::InclusiveSum(
			nullptr,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1,
			pk.m_stream
		);

		if (temp_storage_bytes > CUB_BUFFER_SIZE - 256)
		{
			std::cerr << "Fatal. Not enough CUB_BUFFER. " << temp_storage_bytes + 256 << " < " << CUB_BUFFER_SIZE << "\n";
			exit(1);
		}

		cub::DeviceScan::InclusiveSum(
			d_temp_storage,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1,
			pk.m_stream
		);
	}

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		using RT = typename KC::RT;
		char* result = kc.om.template Pointer<KernelContextT, DynamicStringRequestTag>();
		const uint32_t max_offset = kc.om.template DynamicSize<KernelContextT, DynamicStringRequestTag>();
		uint32_t offset = 0;
		ParsingError err = JsonParse::String(kc, [&](bool& isEscaped, int& activeThreads) {
			uint32_t worker_offset = offset + RT::WorkerId();
			char c = RT::WorkerId() < activeThreads ? kc.wgr.CurrentChar() : '\0';
			if (worker_offset < max_offset)
				result[worker_offset] = c;
			offset += activeThreads;
			return ParsingError::None;
		});
		if (err != ParsingError::None)
			return err;
		kc.om.template Get<KernelContextT, LengthRequestTag>() = offset < max_offset ? offset : max_offset;
		while (offset < max_offset)
		{
			uint32_t worker_offset = offset + RT::WorkerId();
			if (worker_offset < max_offset)
				result[worker_offset] = '\0';
			offset += RT::GroupSize();
		}
		return ParsingError::None;
	}
};

template<class ParserConfigurationT>
__global__ void __launch_bounds__(1024, 2)
	g_gather_strings_v1(
		const uint8_t* input,
		const uint32_t* in_positions,
		const uint32_t* out_positions,
		uint8_t* output,
		const uint32_t count
	)
{
	const uint32_t id = threadIdx.y + blockDim.y * blockIdx.x;
	if (id >= count)
		return;

	const uint32_t in_pos = in_positions[id];
	const uint8_t* in_ptr = input + in_pos;

	const uint32_t out_pos = out_positions[id];
	const uint32_t out_pos_next = out_positions[id + 1];
	const uint32_t length = out_pos_next - out_pos;
	uint8_t* out_ptr = output + out_pos;

	for (int i = 0; i < length; ++i, ++in_ptr, ++out_ptr)
		*out_ptr = *in_ptr;
}

template<class ParserConfigurationT>
__global__ void __launch_bounds__(1024, 2)
	g_gather_strings_v2(
		const uint8_t* input,
		const uint32_t* in_positions,
		const uint32_t* out_positions,
		uint8_t* output,
		const uint32_t count
	)
{
	const uint32_t id = threadIdx.y + blockDim.y * blockIdx.x;
	if (id >= count)
		return;

	const uint32_t in_pos = in_positions[id];
	const uint8_t* in_ptr = input + in_pos;

	const uint32_t out_pos = out_positions[id];
	const uint32_t out_pos_next = out_positions[id + 1];
	const uint32_t length = out_pos_next - out_pos;
	uint8_t* out_ptr = output + out_pos;

	in_ptr += threadIdx.x;
	out_ptr += threadIdx.x;
	for (int i = threadIdx.x; i < length; i += 32, in_ptr += 32, out_ptr += 32)
		*out_ptr = *in_ptr;
}

template<class ParserConfigurationT>
__global__ void __launch_bounds__(1024, 2)
	g_gather_strings_v3(
		const uint8_t* input,
		const uint32_t* in_positions,
		const uint32_t* out_positions,
		uint8_t* output,
		const uint32_t count
	)
{
	const uint32_t id = threadIdx.y + blockDim.y * blockIdx.x;
	if (id >= count)
		return;

	const uint32_t in_pos = in_positions[id];
	const uint8_t* in_ptr = input + in_pos;

	const uint32_t out_pos = out_positions[id];
	const uint32_t out_pos_next = out_positions[id + 1];
	const uint32_t length = out_pos_next - out_pos;
	uint8_t* out_ptr = output + out_pos;

	int32_t in_misalignment = static_cast<int32_t>(reinterpret_cast<uint64_t>(in_ptr)) & 0x3;
	int32_t out_misalignment = static_cast<int32_t>(reinterpret_cast<uint64_t>(out_ptr)) & 0x3;

	in_ptr = reinterpret_cast<const uint8_t*>(reinterpret_cast<uint64_t>(in_ptr) & ~(0x3ull)); //alignment to 0x4
	out_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t>(out_ptr) & ~(0x3ull)); //alignment to 0x4

	uint32_t missalignmented_data = 0;
	int32_t copied = 0;

	/*
	if (in_misalignment)
	{
		if (threadIdx.x < 2)
			missalignmented_data = reinterpret_cast<uint32_t*>(in_ptr)[threadIdx.x];
		in_ptr += 8;
		copied += 8 - in_misalignment;
	}
	
	while (copied < length)
	{
		uint32_t valid_data = 128 - (4 - in_misalignment);

		//Load 4 bytes in 32 threads = 128 bytes
		uint32_t data;
		if (copied + threadIdx.x * 8 < length)
		{
			data = reinterpret_cast<uint32_t*>(in_ptr)[threadIdx.x]; // 4 bytes loaded from global memory
		}

		if (in_misalignment)
		{
			uint32_t next_data = cub::ShuffleDown(data, 1, 31, 0xff'ff'ff'ff);
			//test if cub::PRMT will be faster
			data = (data >> 8 * in_misalignment) | (next_data << 8 * (8 - in_misalignment));
			valid_data -= in_misalignment;
		}

		if (out_misalignment)
		{
			uint64_t first_data = cub::ShuffleIndex(data, 0, 0xff'ff'ff'ff);
			if (copied + threadIdx.x < length)
			{
				in_ptr[threadIdx.x] = static_cast<uint8_t>(first_data >> (8 * threadIdx.x));
			}
		}

		in_ptr += 128;
		out_ptr += 128;
		copied += valid_data;
	}
	*/
}

template<class TagT>
struct JStringDynamicCopyV2
{
	using type = JStringDynamicCopyV2<TagT>;
	using Tag = TagT;
	using LengthRequestTag = std::pair<TagT, boost::mp11::mp_int<0>>;
	using LengthRequest = OutputRequest<
		LengthRequestTag,
		uint32_t,
		boost::mp11::mp_list<
			OutputOptElementsBefore_c<1>
		>
	>;
	using DynamicStringRequestTag = std::pair<TagT, boost::mp11::mp_int<1>>;
	using DynamicStringRequest = DynamicOutputRequest<DynamicStringRequestTag, char>;
	using OffsetsRequestTag = std::pair<TagT, boost::mp11::mp_int<2>>;
	using OffsetsRequest = OutputRequest<OffsetsRequestTag, uint32_t, boost::mp11::mp_list<OutputOptHelpBuffer<void>>>;
	using Printer = DynamicStringPrinter<type>;
	using OutputRequests = boost::mp11::mp_list<LengthRequest, DynamicStringRequest, OffsetsRequest>;
	using MemoryRequests = JsonParse::StringRequests;
	using ParserRequirements = boost::mp11::mp_list<ParserRequirement::KeepDistance>;

	//excpect ParserKernel
	template<class PK>
	static void PostKernelHook(
		PK& pk,
		const char* input,
		const InputIndex* indices,
		ParsingError* errors,
		const uint32_t count,
		void** h_outputs
	)
	{
		using OM = OutputManager<typename PK::BaseAction>;

		uint32_t* lengths = reinterpret_cast<uint32_t*>(
			h_outputs[OM::template TagIndex<LengthRequestTag>::value]
		);
		uint32_t* offsets = reinterpret_cast<uint32_t*>(
			h_outputs[OM::template TagIndex<OffsetsRequestTag>::value]
		);
		char* content = reinterpret_cast<char*>(
			h_outputs[OM::template TagIndex<DynamicStringRequestTag>::value]
		);

		const size_t CUB_BUFFER_SIZE = PK::CUB_BUFFER_SIZE;

		uint8_t* d_temp_storage = pk.m_cub_buffer;
		size_t temp_storage_bytes = 0;

		thrust::transform(
			thrust::cuda::par.on(pk.m_stream),
			indices,
			indices + count,
			offsets,
			offsets,
			thrust::plus<uint32_t>()
		);

		cub::DeviceScan::InclusiveSum(
			nullptr,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1,
			pk.m_stream
		);

		if (temp_storage_bytes > CUB_BUFFER_SIZE)
		{
			std::cerr << "Fatal. Not enough CUB_BUFFER. " << temp_storage_bytes << " < " << CUB_BUFFER_SIZE << "\n";
			exit(1);
		}

		cub::DeviceScan::InclusiveSum(
			d_temp_storage,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1,
			pk.m_stream
		);

		const int block = 1024;

		//const int group = 1;
		//KernelLauncher(&g_gather_strings_v1<typename PK::PC>)
		//  ((count * group + 1024) / 1024, { group, block / group, 1 }, 0, pk.m_stream)
		//  (reinterpret_cast<const uint8_t*>(input), offsets, lengths, reinterpret_cast<uint8_t*>(content), count);

		const int group = 32;
		KernelLauncher(&g_gather_strings_v2<typename PK::PC>)
		  ((count * group + 1024) / 1024, { group, block / group, 1 }, 0, pk.m_stream)
		  (reinterpret_cast<const uint8_t*>(input), offsets, lengths, reinterpret_cast<uint8_t*>(content), count);
	}

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		using RT = typename KC::RT;
		const uint32_t max_offset = kc.om.template DynamicSize<KernelContextT, DynamicStringRequestTag>();
		uint32_t offset = 0;
		kc.om.template Get<KernelContextT, OffsetsRequestTag>() = kc.wgr.GroupDistance() + 1; // +1 to skip "
		ParsingError err = JsonParse::String(kc, [&](bool& isEscaped, int& activeThreads) {
			offset += activeThreads;
			return ParsingError::None;
		});
		if (err != ParsingError::None)
			return err;
		kc.om.template Get<KernelContextT, LengthRequestTag>() = offset < max_offset ? offset : max_offset;
		return ParsingError::None;
	}
};

template<class ParserConfigurationT>
__global__ void __launch_bounds__(1024, 2)
	g_gather_strings_even_spaced(
		const uint8_t* input,
		uint32_t space_per_string,
		const uint32_t* out_positions,
		uint8_t* output,
		const uint32_t count
	)
{
	const uint32_t id = threadIdx.y + blockDim.y * blockIdx.x;
	if (id >= count)
		return;

	const uint8_t* in_ptr = input + space_per_string * id;

	const uint32_t out_pos = out_positions[id];
	const uint32_t out_pos_next = out_positions[id + 1];
	const uint32_t length = out_pos_next - out_pos;
	uint8_t* out_ptr = output + out_pos;

	in_ptr += threadIdx.x;
	out_ptr += threadIdx.x;
	for (int i = threadIdx.x; i < length; i += 32, in_ptr += 32, out_ptr += 32)
		*out_ptr = *in_ptr;
}

template<class TagT>
struct JStringDynamicCopyV3
{
	using type = JStringDynamicCopyV3<TagT>;
	using Tag = TagT;
	using LengthRequestTag = std::pair<TagT, boost::mp11::mp_int<0>>;
	using LengthRequest = OutputRequest<
		LengthRequestTag,
		uint32_t,
		boost::mp11::mp_list<
			OutputOptElementsBefore_c<1>
		>
	>;
	using DynamicStringRequestTag = std::pair<TagT, boost::mp11::mp_int<1>>;
	using DynamicStringRequest = DynamicOutputRequest<DynamicStringRequestTag, char>;
	using DynamicStringInternalRequestTag = std::pair<TagT, boost::mp11::mp_int<2>>;
	using DynamicStringInternalRequest = DynamicOutputRequest<DynamicStringInternalRequestTag, char, boost::mp11::mp_list<OutputOptHelpBuffer<void>>>;

	using Printer = DynamicStringPrinter<type>;
	using OutputRequests = boost::mp11::mp_list<LengthRequest, DynamicStringRequest, DynamicStringInternalRequest>;
	using MemoryRequests = JsonParse::StringRequests;

	//excpect ParserKernel
	template<class PK>
	static void PostKernelHook(
		PK& pk,
		const char* input,
		const InputIndex* indices,
		ParsingError* errors,
		const uint32_t count,
		void** h_outputs
	)
	{
		using OM = OutputManager<typename PK::BaseAction>;

		uint32_t* lengths = reinterpret_cast<uint32_t*>(
			h_outputs[OM::template TagIndex<LengthRequestTag>::value]
		);
		char* content = reinterpret_cast<char*>(
			h_outputs[OM::template TagIndex<DynamicStringInternalRequestTag>::value]
		);
		char* out_content = reinterpret_cast<char*>(
			h_outputs[OM::template TagIndex<DynamicStringRequestTag>::value]
		);

		auto dynamic_size = pk.m_launch_config->dynamic_sizes[OM::template DynamicTagIndex<DynamicStringInternalRequestTag>::value];

		const size_t CUB_BUFFER_SIZE = PK::CUB_BUFFER_SIZE;

		uint8_t* d_temp_storage = pk.m_cub_buffer;
		size_t temp_storage_bytes = 0;

		cub::DeviceScan::InclusiveSum(
			nullptr,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1,
			pk.m_stream
		);

		if (temp_storage_bytes > CUB_BUFFER_SIZE)
		{
			std::cerr << "Fatal. Not enough CUB_BUFFER. " << temp_storage_bytes << " < " << CUB_BUFFER_SIZE << "\n";
			exit(1);
		}

		cub::DeviceScan::InclusiveSum(
			d_temp_storage,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1,
			pk.m_stream
		);

		const int block = 1024;

		const int group = 32;
		KernelLauncher(&g_gather_strings_even_spaced<typename PK::PC>)
		  ((count * group + 1024) / 1024, { group, block / group, 1 }, 0, pk.m_stream)
		  (reinterpret_cast<const uint8_t*>(content), dynamic_size, lengths, reinterpret_cast<uint8_t*>(out_content), count);
	}

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		using RT = typename KC::RT;
		char* result = kc.om.template Pointer<KernelContextT, DynamicStringInternalRequestTag>();
		const uint32_t max_offset = kc.om.template DynamicSize<KernelContextT, DynamicStringInternalRequestTag>();
		uint32_t offset = 0;
		ParsingError err = JsonParse::String(kc, [&](bool& isEscaped, int& activeThreads) {
			uint32_t worker_offset = offset + RT::WorkerId();
			char c = RT::WorkerId() < activeThreads ? kc.wgr.CurrentChar() : '\0';
			if (worker_offset < max_offset)
				result[worker_offset] = c;
			offset += activeThreads;
			return ParsingError::None;
		});
		if (err != ParsingError::None)
			return err;
		kc.om.template Get<KernelContextT, LengthRequestTag>() = offset < max_offset ? offset : max_offset;
		while (offset < max_offset)
		{
			uint32_t worker_offset = offset + RT::WorkerId();
			if (worker_offset < max_offset)
				result[worker_offset] = '\0';
			offset += RT::GroupSize();
		}
		return ParsingError::None;
	}
};
