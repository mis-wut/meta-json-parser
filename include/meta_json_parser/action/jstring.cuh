#pragma once
#include <utility>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/static_buffer.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/output_printer.cuh>
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
	static void PostKernelHook(PK& pk, const uint32_t count, void** outputs)
	{
		using OM = OutputManager<typename PK::BaseAction>;

		uint32_t* lengths = reinterpret_cast<uint32_t*>(
			outputs[OM::template TagIndex<LengthRequestTag>::value]
		);
		char* content = reinterpret_cast<char*>(
			outputs[OM::template TagIndex<DynamicStringRequestTag>::value]
		);
		
		auto dynamic_size = pk.m_launch_config->dynamic_sizes[OM::template DynamicTagIndex<DynamicStringRequestTag>::value];

		size_t* d_num_selected_out = nullptr;
		uint8_t* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;

		cudaMalloc(&d_num_selected_out, sizeof(size_t));

		cub::DeviceSelect::If(
			nullptr,
			temp_storage_bytes,
			content,
			content,
			d_num_selected_out,
			count * dynamic_size,
			IsNotNullByte()
		);

		cudaMalloc(&d_temp_storage, temp_storage_bytes);

		cub::DeviceSelect::If(
			d_temp_storage,
			temp_storage_bytes,
			content,
			content,
			d_num_selected_out,
			count * dynamic_size,
			IsNotNullByte()
		);

		size_t old_storage_bytes = temp_storage_bytes;

		cub::DeviceScan::InclusiveSum(
			nullptr,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1
		);

		if (temp_storage_bytes > old_storage_bytes)
		{
			cudaFree(d_temp_storage);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}

		cub::DeviceScan::InclusiveSum(
			d_temp_storage,
			temp_storage_bytes,
			lengths,
			lengths,
			count + 1
		);

		cudaFree(d_temp_storage);
		cudaFree(d_num_selected_out);
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

