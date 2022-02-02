#ifndef META_JSON_PARSER_TEST_CONTEXT_CUH
#define META_JSON_PARSER_TEST_CONTEXT_CUH
#pragma once
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/kernel_launch_configuration.cuh>
#include <meta_json_parser/parsing_error.h>

class TestContext {
protected:
    size_t m_test_size;
    size_t m_group_size;

    thrust::host_vector<char> m_h_input;
    thrust::host_vector<InputIndex> m_h_indices;
    thrust::device_vector<char> m_d_input;
    thrust::device_vector<InputIndex> m_d_indices;
    thrust::device_vector<ParsingError> m_d_errors;

    using RandomGenerator = std::minstd_rand;
    using SeedType = RandomGenerator::result_type;
    RandomGenerator m_rand;

    virtual void OutputValidate() = 0;

    TestContext(size_t test_size, size_t group_size, SeedType seed);
public:
    virtual void Initialize() = 0;

    void Validate();

    virtual void SetupConfiguration(KernelLaunchConfiguration launch_configuration);

    virtual thrust::host_vector<void*> OutputBuffers() = 0;

    thrust::device_vector<char>& InputData();

    thrust::device_vector<InputIndex>& InputIndices();

    thrust::device_vector<ParsingError>& Errors();

    size_t TestSize() const;

    size_t GroupSize() const;
};

#endif //META_JSON_PARSER_TEST_CONTEXT_CUH
