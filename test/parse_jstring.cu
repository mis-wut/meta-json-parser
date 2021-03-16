#include <gtest/gtest.h>
#include <boost/mp11/integral.hpp>
#include <random>
#include <unordered_set>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/action/jstring.cuh>

class ParseJStringTest : public ::testing::Test {
public:
#if _DEBUG
	static constexpr size_t TEST_SIZE = 0x11;
#else
	static constexpr size_t TEST_SIZE = 0x8001;
#endif
};

struct TestContextStringValidation {
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	TestContextStringValidation(size_t testSize, size_t group_size) {
		std::minstd_rand rng;
		std::uniform_int_distribution<uint32_t> r_chars((uint32_t)'A', (uint32_t)'Z');
		const size_t MIN_LEN = group_size - 4;
		const size_t MAX_LEN = group_size + 4;
		const size_t MAX_STR_LEN = MAX_LEN + 3; //" + " + \0
		std::uniform_int_distribution<uint32_t> r_len(MIN_LEN, MAX_LEN);
		h_input = thrust::host_vector<char>(testSize * MAX_STR_LEN);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		*ind_it = 0;
		++ind_it;
		std::vector<char> word(MAX_LEN + 1);
		for (size_t i = 0; i < testSize; ++i)
		{
			auto len = r_len(rng);
			*std::generate_n(word.begin(), len, [&]() { return r_chars(rng); }) = '\0';
			inp_it += snprintf(inp_it, MAX_STR_LEN, "\"%s\"", word.data());
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_indices = thrust::device_vector<InputIndex>(h_indices);
	}
};

struct TestContextStringValidationBackslash {
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	TestContextStringValidationBackslash(size_t testSize, size_t group_size) {
		std::minstd_rand rng;
		std::uniform_int_distribution<uint32_t> r_chars((uint32_t)'A', (uint32_t)'Z');
		const size_t MIN_LEN = group_size - 4;
		const size_t MAX_LEN = group_size + 4;
		const size_t MAX_STR_LEN = MAX_LEN + 3; //" + " + \0
		std::uniform_int_distribution<uint32_t> r_len(MIN_LEN, MAX_LEN);
		h_input = thrust::host_vector<char>(testSize * MAX_STR_LEN);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		*ind_it = 0;
		++ind_it;
		std::vector<char> escapable({'"', '\\', '/', 'b', 'f', 'n', 'r', 't' });
		std::vector<char> word(MAX_LEN + 1);
		for (size_t i = 0; i < testSize; ++i)
		{
			std::unordered_set<uint32_t> used_pos;
			auto len = r_len(rng);
			*std::generate_n(word.begin(), len, [&]() { return r_chars(rng); }) = '\0';
			do
			{
				auto slash = r_len(rng) % (len - 1);
				if (used_pos.find(slash) != used_pos.end() || used_pos.find(slash + 1) != used_pos.end())
					break;
				word[slash] = '\\';
				word[slash + 1] = escapable[r_len(rng) % escapable.size()];
				used_pos.insert(slash);
				used_pos.insert(slash + 1);
			} while (true);
			inp_it += snprintf(inp_it, MAX_STR_LEN, "\"%s\"", word.data());
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_indices = thrust::device_vector<InputIndex>(h_indices);
	}
};

template<int CopyBytes>
struct TestContextStringStaticCopy {
	thrust::host_vector<char> h_input;
	thrust::host_vector<char> h_correct;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<char> d_input;
	thrust::device_vector<char> d_correct;
	thrust::device_vector<InputIndex> d_indices;

	TestContextStringStaticCopy(size_t testSize, size_t group_size) {
		std::minstd_rand rng;
		std::uniform_int_distribution<uint32_t> r_chars((uint32_t)'A', (uint32_t)'Z');
		const size_t MIN_LEN = CopyBytes / 2;
		const size_t MAX_LEN = CopyBytes;
		const size_t MAX_STR_LEN = MAX_LEN + 3; //" + " + \0
		std::uniform_int_distribution<uint32_t> r_len(MIN_LEN, MAX_LEN);
		h_input = thrust::host_vector<char>(testSize * MAX_STR_LEN);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		h_correct = thrust::host_vector<char>(testSize * CopyBytes);
		std::fill(h_correct.begin(), h_correct.end(), '\0');
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		auto out_it = h_correct.data();
		*ind_it = 0;
		++ind_it;
		std::vector<char> escapable({'"', '\\', '/', 'b', 'f', 'n', 'r', 't' });
		std::vector<char> word(MAX_LEN + 1);
		for (size_t i = 0; i < testSize; ++i)
		{
			std::unordered_set<uint32_t> used_pos;
			auto len = r_len(rng);
			*std::generate_n(word.begin(), len, [&]() { return r_chars(rng); }) = '\0';
			do
			{
				auto slash = r_len(rng) % (len - 1);
				if (used_pos.find(slash) != used_pos.end() || used_pos.find(slash + 1) != used_pos.end())
					break;
				word[slash] = '\\';
				word[slash + 1] = escapable[r_len(rng) % escapable.size()];
				used_pos.insert(slash);
				used_pos.insert(slash + 1);
			} while (true);
			snprintf(out_it, MAX_LEN + 1, "%s", word.data());
			out_it += CopyBytes;
			inp_it += snprintf(inp_it, MAX_STR_LEN, "\"%s\"", word.data());
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_indices = thrust::device_vector<InputIndex>(h_indices);
		d_correct = thrust::device_vector<char>(h_correct);
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

template<class TestContext, int GroupSizeT>
void templated_ParseStringValidation(ParseJStringTest &test)
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using MC = EmptyMemoryConfiguration;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using PC = ParserConfiguration<RT, MC>;
	using _Zero = boost::mp11::mp_int<0>;
	using _One = boost::mp11::mp_int<1>;
	using BA = JString;
	using PK = ParserKernel<PC, BA>;
	const size_t INPUT_T = ParseJStringTest::TEST_SIZE;
	TestContext context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::host_vector<void*> h_outputs(0);
	thrust::device_vector<void*> d_outputs(h_outputs);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::Other);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	typename PK::Launcher(&_parser_kernel<PC, BA>)(BLOCKS_COUNT)(
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
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
}

template<int CopyBytes, int GroupSizeT>
void templated_ParseStringStaticCopy(ParseJStringTest &test)
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using MC = EmptyMemoryConfiguration;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using PC = ParserConfiguration<RT, MC>;
	using _Zero = boost::mp11::mp_int<0>;
	using _One = boost::mp11::mp_int<1>;
	using BA = JStringStaticCopy<boost::mp11::mp_int<CopyBytes>, char>;
	using PK = ParserKernel<PC, BA>;
	const size_t INPUT_T = ParseJStringTest::TEST_SIZE;
	TestContextStringStaticCopy<CopyBytes> context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::device_vector<char> d_result(INPUT_T * CopyBytes);
	thrust::host_vector<void*> h_outputs(1);
	h_outputs[0] = d_result.data().get();
	thrust::device_vector<void*> d_outputs(h_outputs);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::Other);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	typename PK::Launcher(&_parser_kernel<PC, BA>)(BLOCKS_COUNT)(
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
	thrust::host_vector<char> h_result(d_result);
	ASSERT_TRUE(thrust::equal(context.d_correct.begin(), context.d_correct.end(), d_result.begin()));
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
}


TEST_F(ParseJStringTest, validation_W32) {
	templated_ParseStringValidation<TestContextStringValidation, 32>(*this);
}

TEST_F(ParseJStringTest, validation_W16) {
	templated_ParseStringValidation<TestContextStringValidation, 16>(*this);
}

TEST_F(ParseJStringTest, validation_W8) {
	templated_ParseStringValidation<TestContextStringValidation, 8>(*this);
}

TEST_F(ParseJStringTest, validation_backslash_W32) {
	templated_ParseStringValidation<TestContextStringValidationBackslash, 32>(*this);
}

TEST_F(ParseJStringTest, validation_backslash_W16) {
	templated_ParseStringValidation<TestContextStringValidationBackslash, 16>(*this);
}

TEST_F(ParseJStringTest, validation_backslash_W8) {
	templated_ParseStringValidation<TestContextStringValidationBackslash, 8>(*this);
}

TEST_F(ParseJStringTest, static_copy_B5_W32) {
	templated_ParseStringStaticCopy<5, 32>(*this);
}

TEST_F(ParseJStringTest, static_copy_B5_W16) {
	templated_ParseStringStaticCopy<5, 16>(*this);
}

TEST_F(ParseJStringTest, static_copy_B5_W8) {
	templated_ParseStringStaticCopy<5, 8>(*this);
}

TEST_F(ParseJStringTest, static_copy_B33_W32) {
	templated_ParseStringStaticCopy<33, 32>(*this);
}

TEST_F(ParseJStringTest, static_copy_B33_W16) {
	templated_ParseStringStaticCopy<33, 16>(*this);
}

TEST_F(ParseJStringTest, static_copy_B33_W8) {
	templated_ParseStringStaticCopy<33, 8>(*this);
}

TEST_F(ParseJStringTest, static_copy_B60_W32) {
	templated_ParseStringStaticCopy<60, 32>(*this);
}

TEST_F(ParseJStringTest, static_copy_B60_W16) {
	templated_ParseStringStaticCopy<60, 16>(*this);
}

TEST_F(ParseJStringTest, static_copy_B60_W8) {
	templated_ParseStringStaticCopy<60, 8>(*this);
}

