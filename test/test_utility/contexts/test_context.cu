#include <algorithm>
#include <memory>
#include <gtest/gtest.h>
#include <thrust/logical.h>
#include "test_context.cuh"
#include "../operators.cuh"

thrust::device_vector<char> &TestContext::InputData() {
    return m_d_input;
}

thrust::device_vector<InputIndex> &TestContext::InputIndices() {
    return m_d_indices;
}

size_t TestContext::TestSize() const {
    return m_test_size;
}

size_t TestContext::GroupSize() const {
    return m_group_size;
}

TestContext::TestContext(size_t test_size, size_t group_size, TestContext::SeedType seed)
    : m_test_size(test_size), m_group_size(group_size), m_rand(seed),
      m_d_errors(thrust::device_vector<ParsingError>(test_size, ParsingError::None))
    { }

void TestContext::Validate() {
    ASSERT_TRUE(cudaGetLastError() == cudaError::cudaSuccess);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
    testing::AssertionResult result = testing::AssertionSuccess();
    auto found = thrust::find_if_not(m_d_errors.begin(), m_d_errors.end(), NoError());
    if (found != m_d_errors.end()) {
        size_t input_id = found - m_d_errors.begin();
        size_t print_len = m_h_indices[input_id + 1] - m_h_indices[input_id];
        constexpr size_t max_print = 64;
        result = testing::AssertionFailure() << "ParsingError at " << input_id << " element. "
                << "Input was \"" <<
                std::string_view(m_h_input.data() + m_h_indices[input_id], std::min(max_print, print_len));
        if (print_len > max_print)
            result << "...";
        result << "\".";
    }
    ASSERT_TRUE(result);
    OutputValidate();
}

thrust::device_vector<ParsingError> &TestContext::Errors() {
    return m_d_errors;
}

void TestContext::SetupConfiguration(KernelLaunchConfiguration launch_configuration) { }
