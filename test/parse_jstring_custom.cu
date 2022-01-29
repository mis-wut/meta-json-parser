#include <random>
#include <algorithm>
#include <boost/mp11/integral.hpp>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <cub/thread/thread_operators.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/action/jstring_custom.cuh>
#include "test_helper.h"

class ParseJStringCustomTest : public ::testing::Test {
public:
#if _DEBUG
    static constexpr size_t TEST_SIZE = 0x11;
#else
    static constexpr size_t TEST_SIZE = 0x8001;
#endif
};

template<char CountCharT>
struct TestContextCountChars {
    thrust::host_vector<char> h_input;
    thrust::host_vector<InputIndex> h_indices;
    thrust::device_vector<char> d_input;
    thrust::device_vector<InputIndex> d_indices;
    thrust::host_vector<uint32_t> h_correct;
    thrust::device_vector<uint32_t> d_correct;

    TestContextCountChars(size_t testSize, size_t group_size) {
        std::minstd_rand rng;
        std::uniform_int_distribution<uint32_t> r_chars((uint32_t)'A', (uint32_t)'Z');
        const size_t MIN_LEN = (group_size + 4 > 0 ? group_size + 4 : 1);
        const size_t MAX_LEN = group_size + 4;
        const size_t MAX_STR_LEN = MAX_LEN + 3; //" + " + \0
        std::uniform_int_distribution<uint32_t> r_len(MIN_LEN, MAX_LEN);
        h_input = thrust::host_vector<char>(testSize * MAX_STR_LEN);
        h_indices = thrust::host_vector<InputIndex>(testSize + 1);
        h_correct = thrust::host_vector<uint32_t>(testSize);
        auto inp_it = h_input.data();
        auto ind_it = h_indices.begin();
        auto cor_it = h_correct.begin();
        *ind_it = 0;
        ++ind_it;
        std::vector<char> word(MAX_LEN + 1);
        for (size_t i = 0; i < testSize; ++i)
        {
            auto len = r_len(rng);
            *std::generate_n(word.begin(), len, [&]() { return r_chars(rng); }) = '\0';
            inp_it += snprintf(inp_it, MAX_STR_LEN, "\"%s\"", word.data());
            *ind_it = (inp_it - h_input.data());
            *(cor_it++) = std::count(word.begin(), word.end(), CountCharT);
            ++ind_it;
        }
        d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
        thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
        d_indices = thrust::device_vector<InputIndex>(h_indices);
        d_correct = thrust::device_vector<uint32_t>(h_correct);
    }
};

template<char CountCharT, class TagT>
struct CharCounterFunctor {
    using type = CharCounterFunctor<CountCharT, TagT>;
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, uint32_t>>;
    using MemoryRequests = boost::mp11::mp_list<ReduceRequest<uint32_t>>;

    uint32_t counter;

    __device__ CharCounterFunctor() : counter(0) {};

    template<class KernelContextT>
    inline __device__ ParsingError operator()(KernelContextT& kc, bool& escaped, int& activeChars) {
        using RT = typename KernelContextT::RT;
        if (!escaped && RT::WorkerId() < activeChars && kc.wgr.CurrentChar() == CountCharT)
            counter += 1;
        return ParsingError::None;
    };

    template<class KernelContextT>
    inline __device__ ParsingError finalize(KernelContextT& kc) {
        using RT = typename KernelContextT::RT;
        uint32_t result = Reduce<uint32_t, RT::WorkGroupSize>(kc).Reduce(counter, cub::Sum());
        if (RT::WorkerId() == 0)
            kc.om.template Get<KernelContextT, TagT>() = result;
        return ParsingError::None;
    };

};

struct no_error {
    typedef bool result_type;
    typedef ParsingError argument_type;

    __host__ __device__ bool operator()(const ParsingError& err)
    {
        return err == ParsingError::None;
    }
};

template<int GroupSizeT>
void templated_StringCountChars()
{
    using GroupSize = boost::mp11::mp_int<GroupSizeT>;
    constexpr char COUNT_CHAR = 'A';
    constexpr int GROUP_SIZE = GroupSizeT;
    constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
    using GroupCount = boost::mp11::mp_int<GROUP_COUNT>;
    using RT = RuntimeConfiguration<GroupSize, GroupCount>;
    using BA = JStringCustom<CharCounterFunctor<COUNT_CHAR, char>>;
    using PC = ParserConfiguration<RT, BA>;
    using PK = ParserKernel<PC>;
    const size_t INPUT_T = ParseJStringCustomTest::TEST_SIZE;
    TestContextCountChars<COUNT_CHAR> context(INPUT_T, GROUP_SIZE);
    const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
    thrust::device_vector<ParsingError> d_err(INPUT_T);
    thrust::device_vector<uint32_t> d_result(INPUT_T);
    thrust::host_vector<void*> h_outputs(1);
    h_outputs[0] = d_result.data().get();
    thrust::device_vector<void*> d_outputs(h_outputs);
    thrust::fill(d_err.begin(), d_err.end(), ParsingError::Other);
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
    thrust::host_vector<uint32_t> h_result(d_result);
    ASSERT_TRUE(thrust::equal(context.d_correct.begin(), context.d_correct.end(), d_result.begin()));
    ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
}

#define META_jstring_custom_tests(WS)\
TEST_F(ParseJStringCustomTest, CountChar_W##WS) {\
    templated_StringCountChars<WS>();\
}

META_WS_4(META_jstring_custom_tests)
