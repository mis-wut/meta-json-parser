#include <algorithm>
#include <boost/mp11/integral.hpp>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/thread/thread_operators.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/action/jstring_custom.cuh>
#include "test_helper.h"
#include "test_utility/contexts/string_test_context.cuh"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"

class ParseJStringCustomTest : public ::testing::Test { };

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

template<char CountCharT>
class CountCharTestContext : public StringTestContext {
protected:
    thrust::host_vector<uint32_t> m_h_correct;
    thrust::device_vector<uint32_t> m_d_correct;
    thrust::device_vector<uint32_t> m_d_result;

    void InsertedWordCallback(size_t index, std::string_view word) override {
        m_h_correct[index] = std::count(word.begin(), word.end(), CountCharT);
    }

    void OutputValidate() override {
        ASSERT_TRUE(thrust::equal(m_d_correct.begin(), m_d_correct.end(), m_d_result.begin()));
    }
public:
    CountCharTestContext(size_t testSize, size_t groupSize, SeedType seed)
    : StringTestContext(testSize, groupSize, seed) { }

    thrust::host_vector<void *> OutputBuffers() override {
        thrust::host_vector<void *> result(1);
        result[0] = m_d_result.data().get();
        return result;
    }

    void Initialize() override {
        m_h_correct = thrust::host_vector<uint32_t>(TestSize());
        m_d_result = thrust::device_vector<uint32_t>(TestSize());
        std::fill(m_h_correct.begin(), m_h_correct.end(), 0);
        StringTestContext::Initialize();
        m_d_correct = thrust::device_vector<char>(m_h_correct);
    }
};

template<int GroupSizeT>
void templated_StringCountChars() {
    constexpr char CountChar = 'A';
    CountCharTestContext<CountChar> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetMinimumLength(1);
    context.SetMaximumLength(65);
    context.SetMaximumEscapedCharacters(8);
    context.Initialize();
    using BaseAction = JStringCustom<CharCounterFunctor<CountChar, char>>;
    LaunchTest<BaseAction, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_jstring_custom_tests(WS)\
TEST_F(ParseJStringCustomTest, CountChar_W##WS) {\
    templated_StringCountChars<WS>();\
}

META_WS_4(META_jstring_custom_tests)
