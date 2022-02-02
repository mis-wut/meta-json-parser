#include <gtest/gtest.h>
#include "static_string_test_context.cuh"

void StaticStringTestContext::Initialize() {
    m_h_correct = thrust::host_vector<char>(TestSize() * m_static_string_size);
    m_d_result = thrust::device_vector<char>(TestSize() * m_static_string_size);
    std::fill(m_h_correct.begin(), m_h_correct.end(), '\0');
    StringTestContext::Initialize();
    m_d_correct = thrust::device_vector<char>(m_h_correct);
}

StaticStringTestContext::StaticStringTestContext(
    size_t test_size, size_t group_size, TestContext::SeedType seed, uint32_t static_string_size
) : StringTestContext(test_size, group_size, seed), m_static_string_size(static_string_size) {}

void StaticStringTestContext::InsertedWordCallback(size_t index, std::string_view word) {
    snprintf(
        m_h_correct.data() + index * m_static_string_size,
        m_static_string_size + 1,
        "%s",
        word.data()
    );
}

void StaticStringTestContext::OutputValidate() {
    testing::AssertionResult result = testing::AssertionSuccess();
    if (!thrust::equal(m_d_correct.begin(), m_d_correct.end(), m_d_result.begin())) {
        thrust::host_vector<char> h_result(m_d_result);
        auto mismatch = thrust::mismatch(m_h_correct.begin(), m_h_correct.end(), h_result.end());
        size_t input_id = (mismatch.first - m_h_correct.begin()) / m_static_string_size;
        size_t offset = input_id * m_static_string_size;
        std::string_view expected_str(m_h_correct.data() + offset, m_static_string_size);
        std::string_view result_str(h_result.data() + offset, m_static_string_size);
        result = testing::AssertionFailure()
                << "Mismatch output at " << input_id << " string. "
                << "Expected string \"" << expected_str << "\", result string \"" << result_str << "\".";
    }
    ASSERT_TRUE(result);
}

thrust::host_vector<void *> StaticStringTestContext::OutputBuffers() {
    thrust::host_vector<void *> result(1);
    result[0] = m_d_result.data().get();
    return result;
}
