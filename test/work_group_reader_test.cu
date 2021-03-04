#include <vector>
#include <string>
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

constexpr int MAX_SENTENCE = 512;

class WorkGroupReaderTest : public ::testing::Test {
};

template<template <class...> class WorkerGroupT, class GroupSizeT, class GroupCountT>
__global__ void __launch_bounds__(1024, 2)
	work_group_reader_test(
		const char* input,
		const InputIndex* indices,
		char* output,
		const uint32_t count
	)
{
	using WGR = WorkerGroupT<GroupSizeT>;
	using RT = RuntimeConfiguration<GroupSizeT, GroupCountT>;
	using PC = ParserConfiguration<RT>;
	using KC = KernelContext<PC, OutputConfiguration<boost::mp11::mp_list<>>>;
	__shared__ typename KC::M3::SharedBuffers sharedBuffers;
	KC context(sharedBuffers, input, indices, nullptr);
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

template<template <class ...> class WorkerGroupT, int GroupSizeT>
void templated_ProperDataReading()
{
	constexpr int GROUP_SIZE = GroupSizeT;
	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
	constexpr int SENTENCE_COUNT = 33;
	constexpr int BLOCKS_COUNT = (SENTENCE_COUNT + GROUP_SIZE - 1) / GROUP_SIZE;
	constexpr char sentences[SENTENCE_COUNT][MAX_SENTENCE] = {
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi pulvinar, felis id interdum fringilla, nulla eros malesuada quam, eu tincidunt orci turpis ac lorem. Proin ac posuere eros, ac auctor dolor. Duis vel dui vitae sapien tempor ornare vitae id odio. Vestibulum mattis placerat ",
		"vulputate. Duis sit amet metus turpis. Nullam vehicula ante eu malesuada condimentum. Ut ac congue orci. Sed ac tincidunt nibh. Sed vel augue posuere, dapibus dolor in, fringilla ligula. Nullam pulvinar commodo nunc, a fringilla metus hendrerit nec. In congue vestibulum tempus. In ",
		"commodo congue diam eu commodo. Vestibulum auctor augue ut arcu pulvinar facilisis. Nunc felis lorem, venenatis quis ligula non, venenatis finibus massa. Nulla facilisi. Sed venenatis laoreet tellus at rutrum. Integer porta gravida odio, sed consectetur sem tempus vitae. Donec et ",
		"feugiat ex. Nam id consequat felis, nec accumsan ipsum. Quisque efficitur, nisl eget volutpat vestibulum, eros lacus dapibus orci, eget rutrum mi metus non risus. Aliquam sed molestie nisl. Mauris cursus, justo ac vulputate consequat, turpis tortor aliquam erat, eu volutpat ligula ",
		"purus ac libero. Fusce luctus ipsum lacus, eget fringilla diam sollicitudin eu. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nullam tellus ipsum, viverra a lacus a, dictum luctus purus. Vestibulum ac diam lacinia, luctus velit at, consectetur ",
		"lacus. Morbi pulvinar orci at neque tincidunt, vitae cursus augue aliquet. Etiam viverra commodo fringilla. Mauris lacinia nisi nec dolor convallis, a molestie ipsum rhoncus. Quisque tempor euismod lectus vitae convallis. Cras dignissim, quam sit amet tincidunt ultricies, diam tellus ",
		"viverra erat, ac accumsan diam augue eget leo. In hendrerit sem erat, nec consequat neque porttitor eu. Duis semper massa tellus, sit amet porttitor arcu mattis quis. Sed vehicula ex in arcu ornare cursus. Nam pharetra commodo ligula id aliquam. Quisque eget bibendum libero. Cras ",
		"nec molestie ex. Mauris mollis, ex sed dapibus dapibus, velit metus mattis odio, sit amet fermentum risus nunc in est. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla vitae aliquam lorem, at ullamcorper neque. Sed rhoncus, urna ut pharetra ",
		"semper, mauris diam tincidunt est, ac varius tellus ex eu lacus. Aenean tristique enim quis quam varius tincidunt. Phasellus accumsan mauris nec lectus consectetur, ut interdum felis convallis. Vestibulum imperdiet elementum volutpat. Aliquam diam dui, rutrum ut ante sit amet, dignissim ",
		"sodales magna. Nulla in nisi dapibus, convallis purus in, eleifend dui. Maecenas nunc dolor, posuere non purus at, pharetra finibus justo. Vestibulum iaculis elit eget accumsan venenatis. Sed in volutpat urna, eget ornare lectus. Maecenas non justo et velit vulputate suscipit eget ",
		"non eros. Nulla facilisi. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi pulvinar, felis id interdum fringilla, nulla eros malesuada quam, eu tincidunt orci turpis ac lorem. Proin ac posuere eros, ac auctor dolor. Duis vel dui vitae sapien tempor ornare vitae id odio. ",
		"Vestibulum mattis placerat vulputate. Duis sit amet metus turpis. Nullam vehicula ante eu malesuada condimentum. Ut ac congue orci. Sed ac tincidunt nibh. Sed vel augue posuere, dapibus dolor in, fringilla ligula. Nullam pulvinar commodo nunc, a fringilla metus hendrerit nec. In congue ",
		"vestibulum tempus. In commodo congue diam eu commodo. Vestibulum auctor augue ut arcu pulvinar facilisis. Nunc felis lorem, venenatis quis ligula non, venenatis finibus massa. Nulla facilisi. Sed venenatis laoreet tellus at rutrum. Integer porta gravida odio, sed consectetur sem tempus ",
		"vitae. Donec et feugiat ex. Nam id consequat felis, nec accumsan ipsum. Quisque efficitur, nisl eget volutpat vestibulum, eros lacus dapibus orci, eget rutrum mi metus non risus. Aliquam sed molestie nisl. Mauris cursus, justo ac vulputate consequat, turpis tortor aliquam erat, eu ",
		"volutpat ligula purus ac libero. Fusce luctus ipsum lacus, eget fringilla diam sollicitudin eu. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nullam tellus ipsum, viverra a lacus a, dictum luctus purus. Vestibulum ac diam lacinia, luctus velit ",
		"at, consectetur lacus. Morbi pulvinar orci at neque tincidunt, vitae cursus augue aliquet. Etiam viverra commodo fringilla. Mauris lacinia nisi nec dolor convallis, a molestie ipsum rhoncus. Quisque tempor euismod lectus vitae convallis. Cras dignissim, quam sit amet tincidunt ultricies, ",
		"diam tellus viverra erat, ac accumsan diam augue eget leo. In hendrerit sem erat, nec consequat neque porttitor eu. Duis semper massa tellus, sit amet porttitor arcu mattis quis. Sed vehicula ex in arcu ornare cursus. Nam pharetra commodo ligula id aliquam. Quisque eget bibendum libero. ",
		"Cras nec molestie ex. Mauris mollis, ex sed dapibus dapibus, velit metus mattis odio, sit amet fermentum risus nunc in est. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla vitae aliquam lorem, at ullamcorper neque. Sed rhoncus, urna ut ",
		"pharetra semper, mauris diam tincidunt est, ac varius tellus ex eu lacus. Aenean tristique enim quis quam varius tincidunt. Phasellus accumsan mauris nec lectus consectetur, ut interdum felis convallis. Vestibulum imperdiet elementum volutpat. Aliquam diam dui, rutrum ut ante sit ",
		"amet, dignissim sodales magna. Nulla in nisi dapibus, convallis purus in, eleifend dui. Maecenas nunc dolor, posuere non purus at, pharetra finibus justo. Vestibulum iaculis elit eget accumsan venenatis. Sed in volutpat urna, eget ornare lectus. Maecenas non justo et velit vulputate ",
		"suscipit eget non eros. Nulla facilisi. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi pulvinar, felis id interdum fringilla, nulla eros malesuada quam, eu tincidunt orci turpis ac lorem. Proin ac posuere eros, ac auctor dolor. Duis vel dui vitae sapien tempor ornare ",
		"vitae id odio. Vestibulum mattis placerat vulputate. Duis sit amet metus turpis. Nullam vehicula ante eu malesuada condimentum. Ut ac congue orci. Sed ac tincidunt nibh. Sed vel augue posuere, dapibus dolor in, fringilla ligula. Nullam pulvinar commodo nunc, a fringilla metus hendrerit ",
		"nec. In congue vestibulum tempus. In commodo congue diam eu commodo. Vestibulum auctor augue ut arcu pulvinar facilisis. Nunc felis lorem, venenatis quis ligula non, venenatis finibus massa. Nulla facilisi. Sed venenatis laoreet tellus at rutrum. Integer porta gravida odio, sed consectetur ",
		"sem tempus vitae. Donec et feugiat ex. Nam id consequat felis, nec accumsan ipsum. Quisque efficitur, nisl eget volutpat vestibulum, eros lacus dapibus orci, eget rutrum mi metus non risus. Aliquam sed molestie nisl. Mauris cursus, justo ac vulputate consequat, turpis tortor aliquam ",
		"erat, eu volutpat ligula purus ac libero. Fusce luctus ipsum lacus, eget fringilla diam sollicitudin eu. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nullam tellus ipsum, viverra a lacus a, dictum luctus purus. Vestibulum ac diam lacinia, ",
		"luctus velit at, consectetur lacus. Morbi pulvinar orci at neque tincidunt, vitae cursus augue aliquet. Etiam viverra commodo fringilla. Mauris lacinia nisi nec dolor convallis, a molestie ipsum rhoncus. Quisque tempor euismod lectus vitae convallis. Cras dignissim, quam sit amet ",
		"tincidunt ultricies, diam tellus viverra erat, ac accumsan diam augue eget leo. In hendrerit sem erat, nec consequat neque porttitor eu. Duis semper massa tellus, sit amet porttitor arcu mattis quis. Sed vehicula ex in arcu ornare cursus. Nam pharetra commodo ligula id aliquam. Quisque ",
		"eget bibendum libero. Cras nec molestie ex. Mauris mollis, ex sed dapibus dapibus, velit metus mattis odio, sit amet fermentum risus nunc in est. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla vitae aliquam lorem, at ullamcorper neque. ",
		"Sed rhoncus, urna ut pharetra semper, mauris diam tincidunt est, ac varius tellus ex eu lacus. Aenean tristique enim quis quam varius tincidunt. Phasellus accumsan mauris nec lectus consectetur, ut interdum felis convallis. Vestibulum imperdiet elementum volutpat. Aliquam diam dui, ",
		"rutrum ut ante sit amet, dignissim sodales magna. Nulla in nisi dapibus, convallis purus in, eleifend dui. Maecenas nunc dolor, posuere non purus at, pharetra finibus justo. Vestibulum iaculis elit eget accumsan venenatis. Sed in volutpat urna, eget ornare lectus. Maecenas non justo ",
		"et velit vulputate suscipit eget non eros. Nulla facilisi. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi pulvinar, felis id interdum fringilla, nulla eros malesuada quam, eu tincidunt orci turpis ac lorem. Proin ac posuere eros, ac auctor dolor. Duis vel dui vitae ",
		"sapien tempor ornare vitae id odio. Vestibulum mattis placerat vulputate. Duis sit amet metus turpis. Nullam vehicula ante eu malesuada condimentum. Ut ac congue orci. Sed ac tincidunt nibh. Sed vel augue posuere, dapibus dolor in, fringilla ligula. Nullam pulvinar commodo nunc, a ",
		"fringilla metus hendrerit nec. In congue vestibulum tempus. In commodo congue diam eu commodo. Vestibulum auctor augue ut arcu pulvinar facilisis. Nunc felis lorem, venenatis quis ligula non, venenatis finibus massa. Nulla facilisi. Sed venenatis laoreet tellus at rutrum. Integer "
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
	Launch(work_group_reader_test<WorkerGroupT, boost::mp11::mp_int<GROUP_SIZE>, boost::mp11::mp_int<GROUP_COUNT>>)
		({ BLOCKS_COUNT, 1, 1 }, { GROUP_SIZE, GROUP_COUNT, 1 })
		(d_input.data().get(), d_indices.data().get(), d_result.data().get(), SENTENCE_COUNT);
	ASSERT_TRUE(thrust::equal(d_correct.begin(), d_correct.end(), d_result.begin()));
}

TEST_F(WorkGroupReaderTest, ProperDataReading_W32) {
	templated_ProperDataReading<WorkGroupReader, 32>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_W16) {
	templated_ProperDataReading<WorkGroupReader, 16>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_W8) {
	templated_ProperDataReading<WorkGroupReader, 8>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_Prefetch_W32) {
	templated_ProperDataReading<WorkGroupReaderPrefetch, 32>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_Prefetch_W16) {
	templated_ProperDataReading<WorkGroupReaderPrefetch, 16>();
}

TEST_F(WorkGroupReaderTest, ProperDataReading_Prefetch_W8) {
	templated_ProperDataReading<WorkGroupReaderPrefetch, 8>();
}