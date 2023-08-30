#include <gtest/gtest.h>
#include <boost/mp11/integral.hpp>
#include <random>
#include <string>
#include <unordered_set>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/parser_output_device.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/void_action.cuh>
#include "test_helper.h"

class DynamicOutputTest : public ::testing::TestWithParam<size_t> {
public:
#if _DEBUG
	static constexpr size_t TEST_SIZE = 0x11;
#else
	static constexpr size_t TEST_SIZE = 0x8001;
#endif
};

struct TestContextDynamicStringCopy {
	thrust::host_vector<char> h_input;
	thrust::host_vector<int>  h_correct_offsets;
	thrust::host_vector<char> h_correct_content;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<char> d_input;
	thrust::device_vector<int>  d_correct_offsets;
	thrust::device_vector<char> d_correct_content;
	thrust::device_vector<InputIndex> d_indices;

	TestContextDynamicStringCopy(size_t testSize, size_t group_size, size_t max_str_len) {
		std::minstd_rand rng;
		std::uniform_int_distribution<uint32_t> r_chars((uint32_t)'A', (uint32_t)'Z');
		const size_t MIN_LEN = 1;
		const size_t MAX_LEN = max_str_len;
		const size_t MAX_STR_LEN = MAX_LEN + 3; //" + " + \0
		std::uniform_int_distribution<uint32_t> r_len(MIN_LEN, MAX_LEN);
		h_input = thrust::host_vector<char>(testSize * MAX_STR_LEN);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		h_correct_offsets = thrust::host_vector<int>(testSize + 1);
		h_correct_content = thrust::host_vector<char>();
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		auto off_it = h_correct_offsets.begin();
		auto out_it = std::back_inserter(h_correct_content);
		*off_it = 0;
		*ind_it = 0;
		++ind_it;
		std::vector<char> escapable({'"', '\\', '/', 'b', 'f', 'n', 'r', 't' });
		std::vector<char> word(MAX_LEN + 1);
		for (size_t i = 0; i < testSize; ++i)
		{
			std::unordered_set<uint32_t> used_pos;
			auto len = r_len(rng);
			*std::generate_n(word.begin(), len, [&]() { return r_chars(rng); }) = '\0';
			while (len != 1)
			{
				auto slash = r_len(rng) % (len - 1);
				if (used_pos.find(slash) != used_pos.end() || used_pos.find(slash + 1) != used_pos.end())
					break;
				word[slash] = '\\';
				word[slash + 1] = escapable[r_len(rng) % escapable.size()];
				used_pos.insert(slash);
				used_pos.insert(slash + 1);
			}
			auto offset = *off_it + len;
			++off_it;
			*off_it = offset;

			out_it = std::copy_n(word.begin(), len, out_it);
			inp_it += snprintf(inp_it, MAX_STR_LEN, "\"%s\"", word.data());
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_indices = thrust::device_vector<InputIndex>(h_indices);
		d_correct_offsets = thrust::device_vector<int>(h_correct_offsets);
		d_correct_content = thrust::device_vector<char>(h_correct_content);
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

template<template <class ...> class DynamicStringT, int GroupSizeT>
void templated_DynamicStringCopy(size_t max_str_len)
{
	using GroupSize = boost::mp11::mp_int<GroupSizeT>;
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
	using BA = DynamicStringT<int>;
	using PC = ParserConfiguration<RT, BA>;
	using PK = ParserKernel<PC>;
	using OM = OutputManager<BA>;
	const size_t INPUT_T = DynamicOutputTest::TEST_SIZE;
	TestContextDynamicStringCopy context(INPUT_T, GROUP_SIZE, max_str_len);
	thrust::device_vector<ParsingError> d_err(INPUT_T);
	thrust::fill(d_err.begin(), d_err.end(), ParsingError::Other);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);

	KernelLaunchConfiguration klc;
	klc.dynamic_sizes.push_back(max_str_len);
	if (boost::mp11::mp_similar<JStringDynamicCopyV3<int>, DynamicStringT<int>>::value)
	{
		klc.dynamic_sizes.push_back(max_str_len);
	}
	ParserOutputDevice<BA> output(&klc, INPUT_T);

	thrust::host_vector<void*> h_outputs(output.output_buffers_count);
	auto d_output_it = output.m_d_outputs.begin();
	for (auto& h_output : h_outputs)
		h_output = d_output_it++->data().get();
	thrust::device_vector<void*> d_outputs(h_outputs);

	PK pk(&klc);
	pk.Run(
		context.d_input.data().get(),
		context.d_indices.data().get(),
		nullptr,
		d_err.data().get(),
		d_outputs.data().get(),
		INPUT_T,
		h_outputs.data()
	);
	ASSERT_TRUE(cudaGetLastError() == cudaError::cudaSuccess);
	ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
	auto h_output = output.CopyToHost(pk.m_stream);

	//If doesn't work on release try set 0 to unused length values, h_outputs[LenghtTag][0] might be uninitialized
	ASSERT_TRUE(thrust::equal(
		thrust::device,
		context.d_correct_offsets.begin(),
		context.d_correct_offsets.end(),
		reinterpret_cast<int*>(h_outputs[OM::template TagIndex<typename BA::LengthRequestTag>::value])
	));
	ASSERT_TRUE(thrust::equal(
		thrust::device,
		context.d_correct_content.begin(),
		context.d_correct_content.end(),
		reinterpret_cast<char*>(h_outputs[OM::template TagIndex<typename BA::DynamicStringRequestTag>::value])
	));
	ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
}

#define META_dynamic_string_tests(WS)\
TEST_P(DynamicOutputTest, dynamic_output_copy_string_W##WS) {\
	templated_DynamicStringCopy<JStringDynamicCopy, WS>(GetParam());\
}\
TEST_P(DynamicOutputTest, dynamic_output_copy_v2_string_W##WS) {\
	templated_DynamicStringCopy<JStringDynamicCopyV2, WS>(GetParam());\
}\
TEST_P(DynamicOutputTest, dynamic_output_copy_v3_string_W##WS) {\
	templated_DynamicStringCopy<JStringDynamicCopyV3, WS>(GetParam());\
}

META_WS_4(META_dynamic_string_tests)

INSTANTIATE_TEST_SUITE_P(DynamicString, DynamicOutputTest, testing::Values(6, 18, 42), 
	[](const testing::TestParamInfo<DynamicOutputTest::ParamType>& info) {
		return std::to_string(info.param);
	}
);

