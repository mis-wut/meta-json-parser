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
    : m_test_size(test_size), m_group_size(group_size), m_rand(seed), m_d_errors(test_size, ParsingError::None) { }

void TestContext::Validate() {
    ASSERT_TRUE(cudaGetLastError() == cudaError::cudaSuccess);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
    ASSERT_TRUE(thrust::all_of(m_d_errors.begin(), m_d_errors.end(), NoError()));
    OutputValidate();
}

thrust::device_vector<ParsingError> &TestContext::Errors() {
    return m_d_errors;
}

void TestContext::SetupConfiguration(KernelLaunchConfiguration launch_configuration) { }
