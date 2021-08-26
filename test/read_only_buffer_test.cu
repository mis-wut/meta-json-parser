#include <gtest/gtest.h>
#include <thrust/logical.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/kernel_launch_configuration.cuh>
#include <algorithm>

using namespace boost::mp11;

class ReadOnlyBufferTest : public ::testing::Test {
public:
#if _DEBUG
	static constexpr size_t TEST_SIZE = 0x11;
#else
	static constexpr size_t TEST_SIZE = 0x8001;
#endif
};

template<int Size, class TagT>
struct MockNumbersReadOnlyAction {

	struct NumberFiller {
		using Buffer = StaticBuffer_c<Size>;

		static void __host__ Fill(Buffer& buffer, const KernelLaunchConfiguration* _)
		{
			mp_for_each<mp_iota_c<Size>>([&](auto i) {
				constexpr int I = decltype(i)::value;
				buffer.template Alias<char[Size]>()
					[I] = static_cast<char>('0' + (I % 10));
			});
		}

		static void __host__ FillPtr(char* ptr)
		{
			Fill(*reinterpret_cast<Buffer*>(ptr), nullptr);
		}
	};

	using MemR = FilledMemoryRequest<boost::mp11::mp_int<Size>, NumberFiller, MemoryUsage::ReadOnly, MemoryType::Shared>;
	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, StaticBuffer_c<Size>>>;
	using MemoryRequests = boost::mp11::mp_list<MemR>;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		using RT = typename KernelContextT::RT;
		char (&ro_buffer)[Size] = kc.m3
			.template Receive<MemR>()
			.template Alias<char[Size]>();
		char (&output)[Size] = kc.om
			.template Get<KernelContextT, TagT>()
			.template Alias<char[Size]>();
		int i = RT::WorkerId();
		for (; i < Size; i += RT::GroupSize())
			output[i] = ro_buffer[i];
		return ParsingError::None;
	};
};

template<int Size>
struct TestContextNumberReadOnly {
	thrust::host_vector<char> h_correct;
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<char> d_correct;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	TestContextNumberReadOnly(size_t testSize, size_t group_size)
	{
		using M = MockNumbersReadOnlyAction<Size, void>;
		h_input = thrust::host_vector<char>(1);
		h_correct = thrust::host_vector<char>(testSize * Size);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		std::vector<char> word(Size);
		M::NumberFiller::FillPtr(word.data());
		auto it = h_correct.begin();
		for (int i = 0; i < testSize; ++i)
			it = std::copy(word.begin(), word.end(), it);
		std::fill(h_indices.begin(), h_indices.end(), 0);
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_correct = thrust::device_vector<char>(h_correct);
		d_indices = thrust::device_vector<InputIndex>(h_indices);
	}
};

struct no_error {
	typedef bool result_type;
	typedef ParsingError argument_type;

	__host__ __device__ bool operator()(const ParsingError& err)
	{
		return err == ParsingError::None;
	}
};

template<int NumbersT, int GroupSizeT>
void templated_NumberReadOnly()
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using WGR = WorkGroupReader<GroupSize>;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using BA = MockNumbersReadOnlyAction<NumbersT, char>;
	using PC = ParserConfiguration<RT, BA>;
	using PK = ParserKernel<PC>;
	using M3 = typename PK::M3;
	using BUF = typename M3::ReadOnlyBuffer;
	thrust::host_vector<BUF> h_buff(1);
	M3::FillReadOnlyBuffer(h_buff[0], nullptr);
	const size_t INPUT_T = ReadOnlyBufferTest::TEST_SIZE;
	TestContextNumberReadOnly<NumbersT> context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<BUF> d_buff(h_buff);
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::device_vector<char> d_result(context.d_correct.size());
	thrust::host_vector<void*> h_outputs(1);
	h_outputs[0] = d_result.data().get();
	thrust::device_vector<void*> d_outputs(h_outputs);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::None);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	typename PK::Launcher(&_parser_kernel<PC>)(BLOCKS_COUNT)(
		d_buff.data().get(),
		context.d_input.data().get(),
		context.d_indices.data().get(),
		d_err.data().get(),
		d_outputs.data().get(),
		INPUT_T
	);
	ASSERT_TRUE(cudaGetLastError() == cudaError::cudaSuccess);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	thrust::host_vector<ParsingError> h_err(d_err);
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
	ASSERT_TRUE(thrust::equal(context.d_correct.begin(), context.d_correct.end(), d_result.begin()));
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_5_W32) {
	templated_NumberReadOnly<5, 32>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_5_W16) {
	templated_NumberReadOnly<5, 16>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_5_W8) {
	templated_NumberReadOnly<5, 8>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_29_W32) {
	templated_NumberReadOnly<29, 32>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_29_W16) {
	templated_NumberReadOnly<29, 16>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_29_W8) {
	templated_NumberReadOnly<29, 8>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_67_W32) {
	templated_NumberReadOnly<67, 32>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_67_W16) {
	templated_NumberReadOnly<67, 16>();
}

TEST_F(ReadOnlyBufferTest, readonly_numbers_67_W8) {
	templated_NumberReadOnly<67, 8>();
}

