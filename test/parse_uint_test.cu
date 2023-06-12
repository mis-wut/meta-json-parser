#include <typeinfo>
#include <limits>
#include <algorithm>
#include <random>
#include <memory>
#include <map>
#include <meta_json_parser/config.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/kernel_launcher.cuh>
#include <meta_json_parser/action/void_action.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/equal.h>
#include <thrust/logical.h>
#include "uint_test_context.cuh"
#include "test_helper.h"

template<class GroupSizeT, class GroupCountT, class OutTypeT>
__global__ void __launch_bounds__(1024, 2)
	parse_uint_test(
		const char* input,
		const InputIndex* indices,
		OutTypeT* output,
		ParsingError* error,
		const size_t count
	)
{
	using BaseAction = VoidAction;
	using RT = RuntimeConfiguration<GroupSizeT, GroupCountT>;
	using PC = ParserConfiguration<RT, BaseAction, JsonParse::UnsignedIntegerRequests<OutTypeT>>;
	using PK = ParserKernel<PC>;
	using KC = typename PK::KC;
	__shared__ typename KC::M3::SharedBuffers sharedBuffers;
	KC context(nullptr, sharedBuffers, input, indices, nullptr, nullptr, count);
	if (RT::InputId() >= count)
	{
		return;
	}
	OutTypeT out;
	ParsingError err;
	err = JsonParse::UnsignedInteger<OutTypeT>(context, [&out](auto&& result) {
		out = result;
	});
	if (context.wgr.PeekChar(0) != '\0')
		err = ParsingError::Other;
	if (RT::WorkerId() == 0)
	{
		output[RT::InputId()] = out;
		error[RT::InputId()] = err;
	}
}

class ParseUnsignedIntegerTest : public ::testing::Test {
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
void templated_ParseUnsignedInterger(ParseUnsignedIntegerTest &test)
{
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	const size_t INPUT_T = ParseUnsignedIntegerTest::TEST_SIZE;
	TestContext_u<OutTypeT> context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::device_vector<OutTypeT> d_result(INPUT_T);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::None);
	Launch(parse_uint_test<boost::mp11::mp_int<GROUP_SIZE>, boost::mp11::mp_int<GROUP_COUNT>, OutTypeT>)
		({ BLOCKS_COUNT, 1, 1 }, { GROUP_SIZE, GROUP_COUNT, 1 })(
			context.d_input.data().get(),
			context.d_indices.data().get(),
			d_result.data().get(),
			d_err.data().get(),
			INPUT_T
		);
	thrust::host_vector<ParsingError> h_err(d_err);
	thrust::host_vector<OutTypeT> h_result(d_result);
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
	ASSERT_TRUE(thrust::equal(context.d_correct.begin(), context.d_correct.end(), d_result.begin()));
}

#define META_uint_tests(WS)\
TEST_F(ParseUnsignedIntegerTest, uint32_W##WS) {\
	templated_ParseUnsignedInterger<uint32_t, WS>(*this);\
}\
TEST_F(ParseUnsignedIntegerTest, uint64_W##WS) {\
	templated_ParseUnsignedInterger<uint64_t, WS>(*this);\
}\
TEST_F(ParseUnsignedIntegerTest, uint16_W##WS) {\
	templated_ParseUnsignedInterger<uint16_t, WS>(*this);\
}\
TEST_F(ParseUnsignedIntegerTest, uint8_W##WS) {\
	templated_ParseUnsignedInterger<uint8_t, WS>(*this);\
}

META_WS_4(META_uint_tests)
