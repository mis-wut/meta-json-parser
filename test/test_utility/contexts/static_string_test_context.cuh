#ifndef META_JSON_PARSER_STATIC_STRING_TEST_CONTEXT_CUH
#define META_JSON_PARSER_STATIC_STRING_TEST_CONTEXT_CUH
#include "string_test_context.cuh"


class StaticStringTestContext : public StringTestContext {
protected:
    thrust::host_vector<char> m_h_correct;
    thrust::device_vector<char> m_d_correct;
    thrust::device_vector<char> m_d_result;

    uint32_t m_static_string_size;

    void InsertedWordCallback(size_t index, std::string_view word) override;

    void OutputValidate() override;

public:
    void Initialize() override;

    thrust::host_vector<void *> OutputBuffers() override;

    StaticStringTestContext(
            size_t test_size, size_t group_size, TestContext::SeedType seed, uint32_t static_string_size
    );
};


#endif //META_JSON_PARSER_STATIC_STRING_TEST_CONTEXT_CUH
