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
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/kernel_launcher.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/equal.h>
#include <thrust/logical.h>

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
	using WGR = WorkGroupReader<GroupSizeT>;
	using _MC = MemoryConfiguration<boost::mp11::mp_list<>, boost::mp11::mp_list<>, boost::mp11::mp_list<>>;
	using MC = ExtendRequests<_MC, JsonParse::UnsignedIntegerRequests<OutTypeT, GroupSizeT>>;
	using RT = RuntimeConfiguration<GroupSizeT, GroupCountT>;
	using PC = ParserConfiguration<RT, MC>;
	using KC = KernelContext<PC>;
	__shared__ typename KC::M3::SharedBuffers sharedBuffers;
	KC context(sharedBuffers, input, indices);
	if (RT::InputId() >= count)
	{
		return;
	}
	OutTypeT out;
	ParsingError err;
	err = JsonParse::UnsignedInteger<OutTypeT, GroupSizeT>::KC(context)(out);
	if (context.wgr.PeekChar(0) != '\0')
		err = ParsingError::Other;
	if (RT::WorkerId() == 0)
	{
		output[RT::InputId()] = out;
		error[RT::InputId()] = err;
	}
}

template<class OutTypeT>
struct TestContext {
	thrust::host_vector<OutTypeT> h_correct;
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<OutTypeT> d_correct;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	TestContext(size_t testSize, size_t group_size)
	{
		using GenerateT = boost::mp11::mp_if_c<sizeof(OutTypeT) == 1, uint16_t, OutTypeT>;
		GenerateT MAX_VAL = std::numeric_limits<OutTypeT>::max() - 1;
		size_t MAX_UINT_LEN = (size_t)std::ceil(std::log10((double)MAX_VAL));
		if (MAX_UINT_LEN > group_size - 1)
		{
			MAX_VAL = 1;
			for (int i = 0; i < group_size - 1; ++i)
				MAX_VAL *= 10;
			MAX_VAL -= 1;
			MAX_UINT_LEN = group_size - 1;
		}
		std::minstd_rand rng;
		std::uniform_int_distribution<GenerateT> dist(1, MAX_VAL);
		h_input = thrust::host_vector<char>(testSize * MAX_UINT_LEN);
		h_correct = thrust::host_vector<OutTypeT>(testSize);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		std::generate(h_correct.begin(), h_correct.end(), [&dist, &rng]() { return static_cast<OutTypeT>(dist(rng)); });
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		*ind_it = 0;
		++ind_it;
		for (auto& x : h_correct)
		{
			inp_it += snprintf(inp_it, MAX_UINT_LEN + 1, "%llu", static_cast<uint64_t>(x));
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_correct = thrust::device_vector<OutTypeT>(h_correct);
		d_indices = thrust::device_vector<InputIndex>(h_indices);
	}
};

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
	TestContext<OutTypeT> context(INPUT_T, GROUP_SIZE);
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

TEST_F(ParseUnsignedIntegerTest, uint32_W32) {
	templated_ParseUnsignedInterger<uint32_t, 32>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint32_W16) {
	templated_ParseUnsignedInterger<uint32_t, 16>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint32_W8) {
	templated_ParseUnsignedInterger<uint32_t, 8>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint64_W32) {
	templated_ParseUnsignedInterger<uint64_t, 32>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint64_W16) {
	templated_ParseUnsignedInterger<uint64_t, 16>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint64_W8) {
	templated_ParseUnsignedInterger<uint64_t, 8>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint16_W32) {
	templated_ParseUnsignedInterger<uint16_t, 32>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint16_W16) {
	templated_ParseUnsignedInterger<uint16_t, 16>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint16_W8) {
	templated_ParseUnsignedInterger<uint16_t, 8>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint8_W32) {
	templated_ParseUnsignedInterger<uint8_t, 32>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint8_W16) {
	templated_ParseUnsignedInterger<uint8_t, 16>(*this);
}

TEST_F(ParseUnsignedIntegerTest, uint8_W8) {
	templated_ParseUnsignedInterger<uint8_t, 8>(*this);
}
