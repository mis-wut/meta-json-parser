#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <boost/mp11/integral.hpp>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/action/jstring_custom.cuh>
#include <meta_json_parser/action/string_transform_functors/polynomial_rolling_hash_matcher.cuh>
#include "test_helper.h"
#include "test_utility/contexts/string_test_context.cuh"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"

class ParseJStringHashMatcherTest : public ::testing::Test { };

template<class WorkGroupSize>
uint64_t PolynomialRollingHashSimuluator(uint64_t multiplier, uint64_t modulus, const char* cstr) {
    std::array<uint64_t, WorkGroupSize::value> hash = {};
    std::array<uint64_t, WorkGroupSize::value> power = {};
    std::fill(hash.begin(), hash.end(), 0);
    std::fill(power.begin(), power.end(), 1);
    const auto len = std::strlen(cstr);
    for (int i = 0; i < std::ceil(static_cast<float>(len) / WorkGroupSize::value); ++i) {
        for (int tid = 0; tid < WorkGroupSize::value; ++tid) {
            auto idx = i * WorkGroupSize::value + tid;
            if (idx < len) {
                char c = cstr[idx];
                hash[tid] = (hash[tid] + (c - ' ' + 1) * power[tid]) % modulus;
                power[tid] = (power[tid] * multiplier) % modulus;
            }
        }
    }
    for (int tid = 0; tid < WorkGroupSize::value; ++tid)
        hash[tid] = (hash[tid] * (tid + 1)) % modulus;
    uint64_t result = 0;
    for (int tid = 0; tid < WorkGroupSize::value; ++tid)
        result = (result + hash[tid]) % modulus;
    return result;
}

template<class CategoryT, class MultiplierT, class ModulusT, class WorkGroupSize>
class HashMatcherTestContext : public TestContext  {
protected:
    std::vector<std::string> m_words;
    std::unordered_map<std::string, CategoryT> m_word_mapping;
    thrust::host_vector<CategoryT> m_h_correct;
    thrust::device_vector<CategoryT> m_d_correct;
    thrust::device_vector<CategoryT> m_d_result;

    void OutputValidate() override {
        testing::AssertionResult result = testing::AssertionSuccess();
        if (!thrust::equal(m_d_correct.begin(), m_d_correct.end(), m_d_result.begin())) {
            thrust::host_vector<CategoryT> h_result(m_d_result);
            auto mismatch = thrust::mismatch(m_h_correct.begin(), m_h_correct.end(), h_result.begin());
            size_t input_id = mismatch.first - m_h_correct.begin();
            std::string_view word(m_h_input.data() + m_h_indices[input_id], m_h_indices[input_id + 1] - m_h_indices[input_id]);
            result = testing::AssertionFailure()
                    << "Mismatch output at " << input_id << " input value. "
                    << "Expected category \"" << *mismatch.first << "\", "
                    << "result category \"" << *mismatch.second << "\". "
                    << "Word was \"" << word << "\".";
        }
        ASSERT_TRUE(result);
    }
public:
    HashMatcherTestContext(std::unordered_map<std::string, CategoryT>& word_map, size_t testSize, size_t groupSize, SeedType seed)
            : TestContext(testSize, groupSize, seed), m_word_mapping(word_map), m_words() {
        for (auto& x : m_word_mapping) {
            m_words.push_back(x.first);
        }
    }

    thrust::host_vector<void *> OutputBuffers() override {
        thrust::host_vector<void *> result(1);
        result[0] = m_d_result.data().get();
        return result;
    }

    void Initialize() override {
        m_h_correct = thrust::host_vector<CategoryT>(TestSize());
        m_d_result = thrust::device_vector<CategoryT>(TestSize());
        std::fill(m_h_correct.begin(), m_h_correct.end(), std::numeric_limits<CategoryT>::max());
        std::uniform_int_distribution<uint32_t> r_chars('a', 'z');
        std::uniform_int_distribution<uint32_t> r_words(0, m_words.size() - 1);
        size_t longest_key = 0;
        for (auto& key : m_words) {
            longest_key = std::max(longest_key, key.size());
        }
        std::unordered_map<uint64_t, CategoryT> hashes_to_category;
        for (auto& p : m_word_mapping) {
            hashes_to_category.template insert(
                std::make_pair(
                    PolynomialRollingHashSimuluator<WorkGroupSize>(MultiplierT::value, ModulusT::value, p.first.c_str()),
                    p.second
                )
            );
        }

        const size_t min_len = 1;
        const size_t max_len = std::max(34ul, longest_key);
        const size_t max_str_len = max_len + 3; //" + " + \0
        std::uniform_int_distribution<uint32_t> r_len(min_len, max_len);
        m_h_input = thrust::host_vector<char>(TestSize() * max_str_len);
        m_h_indices = thrust::host_vector<InputIndex>(TestSize() + 1);
        auto inp_it = m_h_input.data();
        auto ind_it = m_h_indices.begin();
        *ind_it = 0;
        ++ind_it;
        std::vector<char> word(max_len + 1);
        for (size_t i = 0; i < TestSize(); ++i)
        {
            if (m_rand() & 0x1) {
                auto& key = m_words[r_words(m_rand)];
                std::strcpy(word.data(), key.c_str());
                m_h_correct[i] = m_word_mapping[key];
            } else {
                auto len = r_len(m_rand);
                *std::generate_n(word.begin(), len, [&]() { return r_chars(m_rand); }) = '\0';
                auto result = static_cast<CategoryT>(0);
                auto found = hashes_to_category.find(
                    PolynomialRollingHashSimuluator<WorkGroupSize>(MultiplierT::value, ModulusT::value, word.data())
                );
                if (found != hashes_to_category.end())
                    // Set result for hash conflicts
                    result = found->second;
                m_h_correct[i] = result;
            }
            inp_it += snprintf(inp_it, max_str_len, "\"%s\"", word.data());
            *ind_it = (inp_it - m_h_input.data());
            ++ind_it;
        }
        m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
        thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
        m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
        m_d_correct = thrust::device_vector<CategoryT>(m_h_correct);
    }
};

template<uint32_t N>
using u32 = std::integral_constant<uint32_t, N>;

using StringMap = boost::mp11::mp_list<
    boost::mp11::mp_list<
        boost::mp11::mp_string<'T', 'e', 's', 't', 'i', 'n', 'g'>,
        u32<1>
    >,
    boost::mp11::mp_list<
        boost::mp11::mp_string<'T', 'e', 's', 't', 'T', 'e', 's', 't', 'T', 'e', 's', 't', 'T', 'e', 's', 't', 'T', 'e', 's', 't'>,
        u32<2>
    >,
    boost::mp11::mp_list<
        boost::mp11::mp_string<'T', 'e', 's', 't', 'X'>,
        u32<3>
    >,
    boost::mp11::mp_list<
        boost::mp11::mp_string<'1', '2'>,
        u32<4>
    >
>;

template<int GroupSizeT>
void templated_PolynomialRollingHash() {
    using Category = uint32_t;
    using Multiplier = std::integral_constant<uint64_t, 31>;
    using Modulus = std::integral_constant<uint64_t, static_cast<uint64_t>(1e9 + 9)>;
    std::unordered_map<std::string, Category> words;
    words[std::string("Testing")] = 1;
    words[std::string("TestTestTestTestTest")] = 2;
    words[std::string("TestX")] = 3;
    words[std::string("12")] = 4;
    HashMatcherTestContext<Category, Multiplier, Modulus, boost::mp11::mp_int<GroupSizeT>> context(words, TEST_SIZE, GroupSizeT, SEED);
    context.Initialize();
    using Functor = PolynomialRollingHashMatcher<Multiplier, Modulus, StringMap, void>;
    using BaseAction = JStringCustom<Functor>;
    LaunchTest<BaseAction, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_jstring_custom_hash_matcher_tests(WS)\
TEST_F(ParseJStringHashMatcherTest, PolynomialRollingHash_W##WS) {\
    templated_PolynomialRollingHash<WS>();\
}

META_WS_4(META_jstring_custom_hash_matcher_tests)
