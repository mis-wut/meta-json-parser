#include <gtest/gtest.h>
#include <thrust/logical.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/mp_string.h>

class ParseJDictTest : public ::testing::Test {
public:
#if _DEBUG
	static constexpr size_t TEST_SIZE = 0x1;
#else
	static constexpr size_t TEST_SIZE = 0x8001;
#endif
};

template<class Key1T, class OutType1T, class Key2T, class OutType2T>
struct TestContextJDict2UInt {
	thrust::host_vector<OutType1T> h_correct_1;
	thrust::host_vector<OutType2T> h_correct_2;
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<OutType1T> d_correct_1;
	thrust::device_vector<OutType2T> d_correct_2;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	TestContextJDict2UInt(size_t testSize, size_t group_size)
	{
		using Generate1T = boost::mp11::mp_if_c<sizeof(OutType1T) == 1, uint16_t, OutType1T>;
		using Generate2T = boost::mp11::mp_if_c<sizeof(OutType1T) == 1, uint16_t, OutType1T>;
		Generate1T MAX_VAL_1 = static_cast<Generate1T>(std::numeric_limits<OutType1T>::max() - 1);
		Generate2T MAX_VAL_2 = static_cast<Generate2T>(std::numeric_limits<OutType2T>::max() - 1);
		size_t MAX_KEY_LEN_1 = boost::mp11::mp_size<Key1T>::value + 2;
		size_t MAX_KEY_LEN_2 = boost::mp11::mp_size<Key2T>::value + 2;
		size_t MAX_UINT_LEN_1 = (size_t)std::ceil(std::log10((double)MAX_VAL_1));
		size_t MAX_UINT_LEN_2 = (size_t)std::ceil(std::log10((double)MAX_VAL_2));
		if (MAX_UINT_LEN_1 > group_size - 1)
		{
			MAX_VAL_1 = 1;
			for (int i = 0; i < group_size - 1; ++i)
				MAX_VAL_1 *= 10;
			MAX_VAL_1 -= 1;
			MAX_UINT_LEN_1 = group_size - 1;
		}
		if (MAX_UINT_LEN_2 > group_size - 1)
		{
			MAX_VAL_2 = 1;
			for (int i = 0; i < group_size - 1; ++i)
				MAX_VAL_2 *= 10;
			MAX_VAL_2 -= 1;
			MAX_UINT_LEN_2 = group_size - 1;
		}
		std::minstd_rand rng;
		std::uniform_int_distribution<Generate1T> dist_1(1, MAX_VAL_1);
		std::uniform_int_distribution<Generate2T> dist_2(1, MAX_VAL_2);
		size_t MAX_LEN = MAX_UINT_LEN_1 + MAX_UINT_LEN_2 + MAX_KEY_LEN_1 + MAX_KEY_LEN_2 + 11;
		h_input = thrust::host_vector<char>(testSize * MAX_LEN);
		h_correct_1 = thrust::host_vector<OutType1T>(testSize);
		h_correct_2 = thrust::host_vector<OutType2T>(testSize);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		std::generate(h_correct_1.begin(), h_correct_1.end(), [&dist_1, &rng]() { return static_cast<OutType1T>(dist_1(rng)); });
		std::generate(h_correct_2.begin(), h_correct_2.end(), [&dist_2, &rng]() { return static_cast<OutType2T>(dist_2(rng)); });
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		*ind_it = 0;
		++ind_it;
		std::vector<char> key_1(boost::mp11::mp_size<Key1T>::value + 1);
		std::vector<char> key_2(boost::mp11::mp_size<Key2T>::value + 1);
		auto key_it = key_1.begin();
		boost::mp11::mp_for_each<Key1T>([&](auto x) {
			*key_it++ = static_cast<char>(decltype(x)::value);
		});
		*key_it = 0;
		key_it = key_2.begin();
		boost::mp11::mp_for_each<Key2T>([&](auto x) {
			*key_it++ = static_cast<char>(decltype(x)::value);
		});
		*key_it = 0;
		for (size_t i = 0; i < testSize; ++i)
		{
			auto x1 = static_cast<long long unsigned int>(h_correct_1[i]);
			auto x2 = static_cast<long long unsigned int>(h_correct_2[i]);
			char* k1 = key_1.data();
			char* k2 = key_2.data();
			if (dist_1(rng) & 0x1)
			{
				std::swap(x1, x2);
				std::swap(k1, k2);
			}
			inp_it += snprintf(inp_it, MAX_LEN + 1, "{ \"%s\": %llu, \"%s\": %llu }", k1, x1, k2, x2);
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_correct_1 = thrust::device_vector<OutType1T>(h_correct_1);
		d_correct_2 = thrust::device_vector<OutType2T>(h_correct_2);
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

template<class OutType1T, class OutType2T, int GroupSizeT>
void templated_ParseDict2UInt(ParseJDictTest &test)
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using WGR = WorkGroupReader<GroupSize>;
	using MC = MemoryConfiguration<boost::mp11::mp_list<>, boost::mp11::mp_list<>, boost::mp11::mp_list<>>;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using PC = ParserConfiguration<RT, MC>;
	using Key1 = boost::mp11::mp_string<'K', 'e', 'y', ' ', 'n', 'u', 'm', ' ', 'o', 'n', 'e'>;
	using Key2 = boost::mp11::mp_string<'S', 'e', 'c', 'o', 'n', 'd', ' ', 'k', 'e', 'y'>;
	using BA = JDict<boost::mp11::mp_list<
		boost::mp11::mp_list<Key1, JNumber<OutType1T, Key1>>,
		boost::mp11::mp_list<Key2, JNumber<OutType2T, Key2>>
	>>;
	using PK = ParserKernel<PC, BA>;
	using M3 = typename PK::M3;
	using BUF = typename M3::ReadOnlyBuffer;
	thrust::host_vector<BUF> h_buff(1);
	M3::FillReadOnlyBuffer(h_buff[0]);
	const size_t INPUT_T = ParseJDictTest::TEST_SIZE;
	TestContextJDict2UInt<Key1, OutType1T, Key2, OutType2T> context(INPUT_T, GROUP_SIZE);
	const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
	thrust::device_vector<BUF> d_buff(h_buff);
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::device_vector<OutType1T> d_result_1(INPUT_T);
	thrust::device_vector<OutType2T> d_result_2(INPUT_T);
	thrust::host_vector<void*> h_outputs(2);
	h_outputs[0] = d_result_1.data().get();
	h_outputs[1] = d_result_2.data().get();
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
	thrust::host_vector<OutType1T> h_result_1(d_result_1);
	thrust::host_vector<OutType2T> h_result_2(d_result_2);
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
	ASSERT_TRUE(thrust::equal(context.d_correct_1.begin(), context.d_correct_1.end(), d_result_1.begin()));
	ASSERT_TRUE(thrust::equal(context.d_correct_2.begin(), context.d_correct_2.end(), d_result_2.begin()));
}

TEST_F(ParseJDictTest, uint8_uint32_W32) {
	templated_ParseDict2UInt<uint8_t, uint32_t, 32>(*this);
}

TEST_F(ParseJDictTest, uint8_uint32_W16) {
	templated_ParseDict2UInt<uint8_t, uint32_t, 16>(*this);
}

TEST_F(ParseJDictTest, uint8_uint32_W8) {
	templated_ParseDict2UInt<uint8_t, uint32_t, 8>(*this);
}

TEST_F(ParseJDictTest, uint64_uint16_W32) {
	templated_ParseDict2UInt<uint64_t, uint16_t, 32>(*this);
}

TEST_F(ParseJDictTest, uint64_uint16_W16) {
	templated_ParseDict2UInt<uint64_t, uint16_t, 16>(*this);
}

TEST_F(ParseJDictTest, uint64_uint16_W8) {
	templated_ParseDict2UInt<uint64_t, uint16_t, 8>(*this);
}

TEST_F(ParseJDictTest, uint64_uint64_W32) {
	templated_ParseDict2UInt<uint64_t, uint64_t, 32>(*this);
}

TEST_F(ParseJDictTest, uint64_uint64_W16) {
	templated_ParseDict2UInt<uint64_t, uint64_t, 16>(*this);
}

TEST_F(ParseJDictTest, uint64_uint64_W8) {
	templated_ParseDict2UInt<uint64_t, uint64_t, 8>(*this);
}

TEST_F(ParseJDictTest, uint8_uint8_W32) {
	templated_ParseDict2UInt<uint8_t, uint8_t, 32>(*this);
}

TEST_F(ParseJDictTest, uint8_uint8_W16) {
	templated_ParseDict2UInt<uint8_t, uint8_t, 16>(*this);
}

TEST_F(ParseJDictTest, uint8_uint8_W8) {
	templated_ParseDict2UInt<uint8_t, uint8_t, 8>(*this);
}

