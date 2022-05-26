//
// Created by lothedr on 21.05.2022.
//

#ifndef META_JSON_PARSER_REPEAT_TEST_CONTEXT_CUH
#define META_JSON_PARSER_REPEAT_TEST_CONTEXT_CUH
#include "test_context.cuh"
#include <vector>
#include <string>

class RepeatTestContext : public TestContext {
protected:
    std::vector<std::string> m_jsons;
    std::vector<uint8_t> m_valid;

    void OutputValidate() override;

public:
    RepeatTestContext(size_t test_size, size_t group_size, SeedType seed);

    void AppendEntry(const std::string& json, bool valid = true);
    void AppendEntry(std::string&& json, bool valid = true);

    void Initialize() override;

    thrust::host_vector<void *> OutputBuffers() override;

};


#endif //META_JSON_PARSER_REPEAT_TEST_CONTEXT_CUH