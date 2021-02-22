#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <gtest/gtest.h>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/kernel_launcher.cuh>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/intelisense_silencer.h>

class WorkGroupReaderTest : public ::testing::Test {
};

constexpr int MAX_SENTENCE = 64;

template<class GroupSizeT, class GroupCountT>
__global__ void __launch_bounds__(1024, 2)
	work_group_reader_test(
		const char* input,
		const InputIndex* indices,
		char* output,
		const uint32_t count
	)
{
	using WGR = WorkGroupReader<GroupSizeT>;
	using MC = MemoryConfiguration<boost::mp11::mp_list<>, boost::mp11::mp_list<>, boost::mp11::mp_list<>>;
	using RT = RuntimeConfiguration<GroupSizeT, GroupCountT>;
	using PC = ParserConfiguration<RT, MC>;
	using KC = KernelContext<PC>;
	__shared__ typename KC::M3::SharedBuffers sharedBuffers;
	KC context(sharedBuffers, input, indices);
	if (RT::InputId() >= count)
	{
		return;
	}
	char* out = reinterpret_cast<char*>(output) + MAX_SENTENCE * RT::InputId() + RT::WorkerId();
	auto wgr = context.wgr;
	while (wgr.PeekChar(0) != '\0')
	{
		char c = wgr.CurrentChar();
		if (c != '\0')
			*out = c;
		out += RT::WorkGroupSize::value;
		wgr.AdvanceBy(RT::WorkGroupSize::value);
	}
}

template<int GroupSizeT>
void templated_ProperDataReading()
{
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	constexpr int SENTENCE_COUNT = 33;
	constexpr int BLOCKS_COUNT = (SENTENCE_COUNT + GROUP_SIZE - 1) / GROUP_SIZE;
	constexpr char sentences[SENTENCE_COUNT][MAX_SENTENCE] = {
		"Lorem ipsum dolor sit amet, ",
		"consectetur adipiscing elit, sed do ",
		"eiusmod tempor incididunt ut labore ",
		"et dolore magna aliqua. Ut ",
		"enim ad minim veniam, quis ",
		"nostrud exercitation ullamco laboris nisi ",
		"ut aliquip ex ea commodo ",
		"consequat. Duis aute irure dolor ",
		"in reprehenderit in voluptate velit ",
		"esse cillum dolore eu fugiat ",
		"nulla pariatur. Excepteur sint occaecat ",
		"cupidatat non proident, sunt in ",
		"culpa qui officia deserunt mollit ",
		"anim id est laborum. ",
		"Lorem ipsum dolor sit amet, ",
		"consectetur adipiscing elit, sed do ",
		"eiusmod tempor incididunt ut labore ",
		"et dolore magna aliqua. Ut ",
		"enim ad minim veniam, quis ",
		"nostrud exercitation ullamco laboris nisi ",
		"ut aliquip ex ea commodo ",
		"consequat. Duis aute irure dolor ",
		"in reprehenderit in voluptate velit ",
		"esse cillum dolore eu fugiat ",
		"nulla pariatur. Excepteur sint occaecat ",
		"cupidatat non proident, sunt in ",
		"culpa qui officia deserunt mollit ",
		"anim id est laborum. ",
		"Lorem ipsum dolor sit amet, ",
		"consectetur adipiscing elit, sed do ",
		"eiusmod tempor incididunt ut labore ",
		"et dolore magna aliqua. Ut ",
		"enim ad minim veniam, quis "
	};
	thrust::host_vector<char> h_input(sizeof(sentences));
	thrust::host_vector<char> h_correct(sizeof(sentences));
	thrust::host_vector<InputIndex> h_indices(SENTENCE_COUNT + 1);
	thrust::fill(h_input.begin(), h_input.end(), '_');
	thrust::fill(h_correct.begin(), h_correct.end(), '%');
	auto h_input_it = h_input.data();
	auto h_correct_it = h_correct.data();
	auto h_indices_it = h_indices.data();
	*h_indices_it = 0;
	++h_indices_it;
	for (int i_sentence = 0; i_sentence < SENTENCE_COUNT; ++i_sentence)
	{
		const char* const sentence = sentences[i_sentence];
		*(h_correct_it + std::snprintf(h_correct_it, MAX_SENTENCE, sentence)) = '%';
		h_correct_it += MAX_SENTENCE;
		h_input_it += std::snprintf(h_input_it, MAX_SENTENCE - 2ull, sentence);
		*h_indices_it = h_input_it - h_input.data();
		++h_indices_it;
	}
	thrust::device_vector<char> d_input(h_input);
	thrust::device_vector<char> d_result(h_input.size());
	thrust::device_vector<char> d_correct(h_correct);
	thrust::device_vector<InputIndex> d_indices(h_indices);
	thrust::fill(d_result.begin(), d_result.end(), '%');
	Launch(work_group_reader_test<boost::mp11::mp_int<GROUP_SIZE>, boost::mp11::mp_int<GROUP_COUNT>>)
		({ BLOCKS_COUNT, 1, 1 }, { GROUP_SIZE, GROUP_COUNT, 1 })
		(d_input.data().get(), d_indices.data().get(), d_result.data().get(), SENTENCE_COUNT);
	ASSERT_TRUE(thrust::equal(d_correct.begin(), d_correct.end(), d_result.begin()));
}

TEST_F(WorkGroupReaderTest, ProperDataReading_32) {
	templated_ProperDataReading<32>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_16) {
	templated_ProperDataReading<16>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_8) {
	templated_ProperDataReading<8>();
}