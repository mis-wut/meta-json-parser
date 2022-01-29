#ifndef META_JSON_PARSER_STRING_TEST_CONTEXT_CUH
#define META_JSON_PARSER_STRING_TEST_CONTEXT_CUH
#include <vector>
#include <numeric>
#include "test_context.cuh"

class StringTestContext : public TestContext {
    std::vector<char> m_allowed_chars;
    std::vector<char> m_allowed_escaped_chars;
    uint32_t m_min_length;
    uint32_t m_max_length;
    uint32_t m_max_escaped;
protected:
    virtual void InsertedWordCallback(const std::vector<char>& word);

    void OutputValidate() override;
public:
    StringTestContext(size_t test_size, size_t group_size, SeedType seed);
    /**
     * Characters included in random generated input.
     */
    const std::vector<char>& GetAllowedChars() const;
    void SetAllowedChars(std::vector<char>& allowed_chars);
    void SetAllowedChars(std::vector<char>&& allowed_chars);

    /**
     * Characters that will be included as escaped characters. They should be escapable.
     */
    const std::vector<char>& GetAllowedEscapedChars() const;
    void SetAllowedEscapedChars(std::vector<char>& allowed_escaped_chars);
    void SetAllowedEscapedChars(std::vector<char>&& allowed_escaped_chars);

    /**
     * Minimum length of random generated string.
     */
    uint32_t GetMinimumLength() const;
    void SetMinimumLength(uint32_t min_len);

    /**
     * Maximum length of random generated string.
     */
    uint32_t GetMaximumLength() const;
    void SetMaximumLength(uint32_t max_len);

    /**
     * Maximum number of escaped characters. It is the upper limit, not every generated string fill have that many
     * escaped characters. Value will be considered only if GetAllowedEscapedChars isn't empty.
     */
    uint32_t GetMaximumEscapedCharacters() const;
    void SetMaximumEscapedCharacters(uint32_t max_escaped);

    thrust::host_vector<void *> OutputBuffers() override;

    void Initialize() override;
};

#endif //META_JSON_PARSER_STRING_TEST_CONTEXT_CUH
