#include <gtest/gtest.h>
#include <random>
#include <thrust/logical.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jbool.cuh>
#include <meta_json_parser/parser_kernel.cuh>

class ParseJBoolTest : public ::testing::Test {
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

template <class T>
struct as_bool : public thrust::unary_function<T, bool>
{
	__host__ __device__ bool operator()(T& x) const
	{
		return static_cast<bool>(x);
	}
};

template<class OutTypeT>
struct BoolTestContext {
	thrust::host_vector<OutTypeT> h_correct;
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<OutTypeT> d_correct;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	BoolTestContext(size_t testSize, size_t group_size)
	{
		using GenerateT = boost::mp11::mp_if_c<sizeof(OutTypeT) == 1, uint16_t, OutTypeT>;
		std::minstd_rand rng;
		std::uniform_int_distribution<GenerateT> dist(0, 1);
		h_input = thrust::host_vector<char>(testSize * 5 + 3);
		h_correct = thrust::host_vector<OutTypeT>(testSize);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		std::generate(h_correct.begin(), h_correct.end(), [&dist, &rng]() { return static_cast<OutTypeT>(dist(rng)); });
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		*ind_it = 0;
		++ind_it;
		auto _true = "true";
		auto _false = "false";
		for (auto& x : h_correct)
		{
			const char* p = x ? _true : _false;
			inp_it += snprintf(inp_it, 5 + 1, "%s", p);
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_correct = thrust::device_vector<OutTypeT>(h_correct);
		d_indices = thrust::device_vector<InputIndex>(h_indices);
	}
};

template<class OutTypeT, int GroupSizeT>
void templated_ParseBool(ParseJBoolTest &test)
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using WGR = WorkGroupReader<GroupSize>;
	using MC = MemoryConfiguration<boost::mp11::mp_list<>, boost::mp11::mp_list<>, boost::mp11::mp_list<>>;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using PC = ParserConfiguration<RT, MC>;
	using BA = JBool<OutTypeT, void>;
	using PK = ParserKernel<PC, BA>;
	using M3 = typename PK::M3;
	using BUF = typename M3::ReadOnlyBuffer;
	thrust::host_vector<BUF> h_buff(1);
	M3::FillReadOnlyBuffer(h_buff[0]);
	const size_t INPUT_T = ParseJBoolTest::TEST_SIZE;
	BoolTestContext<OutTypeT> context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<BUF> d_buff(h_buff);
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::device_vector<OutTypeT> d_result(INPUT_T);
	thrust::host_vector<void*> h_outputs(1);
	h_outputs[0] = d_result.data().get();
	thrust::device_vector<void*> d_outputs(h_outputs);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::None);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	typename PK::Launcher(&_parser_kernel<PC, BA>)(BLOCKS_COUNT)(
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
	thrust::host_vector<OutTypeT> h_result(d_result);
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
	auto correct_begin = thrust::make_transform_iterator(context.d_correct.begin(), as_bool<OutTypeT>());
	auto correct_end = thrust::make_transform_iterator(context.d_correct.end(), as_bool<OutTypeT>());
	auto result_begin = thrust::make_transform_iterator(d_result.begin(), as_bool<OutTypeT>());
	ASSERT_TRUE(thrust::equal(correct_begin, correct_end, result_begin));
}

TEST_F(ParseJBoolTest, bool_uint32_W32) {
	templated_ParseBool<uint32_t, 32>(*this);
}

TEST_F(ParseJBoolTest, bool_uint32_W16) {
	templated_ParseBool<uint32_t, 16>(*this);
}

TEST_F(ParseJBoolTest, bool_uint32_W8) {
	templated_ParseBool<uint32_t, 8>(*this);
}

TEST_F(ParseJBoolTest, bool_uint64_W32) {
	templated_ParseBool<uint64_t, 32>(*this);
}

TEST_F(ParseJBoolTest, bool_uint64_W16) {
	templated_ParseBool<uint64_t, 16>(*this);
}

TEST_F(ParseJBoolTest, bool_uint64_W8) {
	templated_ParseBool<uint64_t, 8>(*this);
}

TEST_F(ParseJBoolTest, bool_uint8_W32) {
	templated_ParseBool<uint8_t, 32>(*this);
}

TEST_F(ParseJBoolTest, bool_uint8_W16) {
	templated_ParseBool<uint8_t, 16>(*this);
}

TEST_F(ParseJBoolTest, bool_uint8_W8) {
	templated_ParseBool<uint8_t, 8>(*this);
}


