#include <gtest/gtest.h>
#include <thrust/logical.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/parser_kernel.cuh>
#include "uint_test_context.cuh"

class ParseJNumberTest : public ::testing::Test {
public:
	static constexpr size_t TEST_SIZE = 0x8000;
};

struct no_error {
	typedef bool result_type;
	typedef ParsingError argument_type;

	__host__ __device__ bool operator()(const ParsingError& err)
	{
		return err == ParsingError::None;
	}
};

template<class OutTypeT, int GroupSizeT>
void templated_ParseUnsignedInterger(ParseJNumberTest &test)
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using BA = JNumber<OutTypeT, void>;
	using PC = ParserConfiguration<RT, BA>;
	using PK = ParserKernel<PC>;
	const size_t INPUT_T = ParseJNumberTest::TEST_SIZE;
	TestContext<OutTypeT> context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::device_vector<OutTypeT> d_result(INPUT_T);
	thrust::host_vector<void*> h_outputs(1);
	h_outputs[0] = d_result.data().get();
	thrust::device_vector<void*> d_outputs(h_outputs);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::None);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	typename PK::Launcher(&_parser_kernel<PC>)(BLOCKS_COUNT)(
		nullptr,
		context.d_input.data().get(),
		context.d_indices.data().get(),
		d_err.data().get(),
		d_outputs.data().get(),
		INPUT_T
	);
	ASSERT_TRUE(cudaGetLastError() == cudaError::cudaSuccess);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	thrust::host_vector<ParsingError> h_err(d_err);
	thrust::host_vector<OutTypeT> h_result(d_result);
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
	ASSERT_TRUE(thrust::equal(context.d_correct.begin(), context.d_correct.end(), d_result.begin()));
}

TEST_F(ParseJNumberTest, uint32_W32) {
	templated_ParseUnsignedInterger<uint32_t, 32>(*this);
}

TEST_F(ParseJNumberTest, uint32_W16) {
	templated_ParseUnsignedInterger<uint32_t, 16>(*this);
}

TEST_F(ParseJNumberTest, uint32_W8) {
	templated_ParseUnsignedInterger<uint32_t, 8>(*this);
}

TEST_F(ParseJNumberTest, uint64_W32) {
	templated_ParseUnsignedInterger<uint64_t, 32>(*this);
}

TEST_F(ParseJNumberTest, uint64_W16) {
	templated_ParseUnsignedInterger<uint64_t, 16>(*this);
}

TEST_F(ParseJNumberTest, uint64_W8) {
	templated_ParseUnsignedInterger<uint64_t, 8>(*this);
}

TEST_F(ParseJNumberTest, uint16_W32) {
	templated_ParseUnsignedInterger<uint16_t, 32>(*this);
}

TEST_F(ParseJNumberTest, uint16_W16) {
	templated_ParseUnsignedInterger<uint16_t, 16>(*this);
}

TEST_F(ParseJNumberTest, uint16_W8) {
	templated_ParseUnsignedInterger<uint16_t, 8>(*this);
}

TEST_F(ParseJNumberTest, uint8_W32) {
	templated_ParseUnsignedInterger<uint8_t, 32>(*this);
}

TEST_F(ParseJNumberTest, uint8_W16) {
	templated_ParseUnsignedInterger<uint8_t, 16>(*this);
}

TEST_F(ParseJNumberTest, uint8_W8) {
	templated_ParseUnsignedInterger<uint8_t, 8>(*this);
}
