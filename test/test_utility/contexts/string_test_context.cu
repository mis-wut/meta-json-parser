#include <unordered_set>
#include "string_test_context.cuh"

const std::vector<char>& StringTestContext::GetAllowedChars() const {
    return m_allowed_chars;
}

const std::vector<char>& StringTestContext::GetAllowedEscapedChars() const {
    return m_allowed_escaped_chars;
}

uint32_t StringTestContext::GetMinimumLength() const {
    return m_min_length;
}

uint32_t StringTestContext::GetMaximumLength() const {
    return m_max_length;
}

uint32_t StringTestContext::GetMaximumEscapedCharacters() const {
    return m_max_escaped;
}

void StringTestContext::Initialize() {
    std::vector<char> chars = GetAllowedChars();
    std::uniform_int_distribution<uint32_t> r_chars(0, chars.size() - 1);
    const size_t min_len = GetMinimumLength();
    const size_t max_len = GetMaximumLength();
    const size_t max_str_len = max_len + 3; //" + " + \0
    std::uniform_int_distribution<uint32_t> r_len(min_len, max_len);
    m_h_input = thrust::host_vector<char>(m_test_size * max_str_len);
    m_h_indices = thrust::host_vector<InputIndex>(m_test_size + 1);
    auto inp_it = m_h_input.data();
    auto ind_it = m_h_indices.begin();
    *ind_it = 0;
    ++ind_it;
    std::vector<char> escapable = GetAllowedEscapedChars();
    std::vector<char> word(max_len + 1);
    for (size_t i = 0; i < m_test_size; ++i)
    {
        std::unordered_set<uint32_t> used_pos;
        auto len = r_len(m_rand);
        *std::generate_n(word.begin(), len, [&]() { return chars[r_chars(m_rand)]; }) = '\0';
        if (len > 1 && !escapable.empty()) {
            for (int j = static_cast<int>(GetMaximumEscapedCharacters()); j > 0 ; --j) {
                auto slash = r_len(m_rand) % (len - 1);
                if (used_pos.find(slash) != used_pos.end() || used_pos.find(slash + 1) != used_pos.end())
                    continue;
                word[slash] = '\\';
                word[slash + 1] = escapable[r_len(m_rand) % escapable.size()];
                used_pos.insert(slash);
                used_pos.insert(slash + 1);
            }
        }
        InsertedWordCallback(word);
        inp_it += snprintf(inp_it, max_str_len, "\"%s\"", word.data());
        *ind_it = (inp_it - m_h_input.data());
        ++ind_it;
    }
    m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
    thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
    m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
}

StringTestContext::StringTestContext(size_t test_size, size_t group_size, TestContext::SeedType seed)
        : TestContext(test_size, group_size, seed),
        m_allowed_escaped_chars(std::vector<char>({'"', '\\', '/', 'b', 'f', 'n', 'r', 't' })),
        m_allowed_chars('Z' - 'A' + 1),
        m_min_length(1),
        m_max_length(64),
        m_max_escaped(0) {
    std::iota(m_allowed_chars.begin(), m_allowed_chars.end(), 'A');
}

void StringTestContext::SetAllowedChars(std::vector<char> &allowed_chars) {
    m_allowed_chars = std::vector<char>(allowed_chars);
}

void StringTestContext::SetAllowedChars(std::vector<char> &&allowed_chars) {
    m_allowed_chars = allowed_chars;
}

void StringTestContext::SetAllowedEscapedChars(std::vector<char> &allowed_escaped_chars) {
    m_allowed_escaped_chars = std::vector<char>(allowed_escaped_chars);
}

void StringTestContext::SetAllowedEscapedChars(std::vector<char> &&allowed_escaped_chars) {
    m_allowed_escaped_chars = allowed_escaped_chars;
}

void StringTestContext::SetMinimumLength(uint32_t min_len) {
    m_min_length = min_len;
}

void StringTestContext::SetMaximumLength(uint32_t max_len) {
    m_max_length = max_len;
}

void StringTestContext::SetMaximumEscapedCharacters(uint32_t max_escaped) {
    m_max_escaped = max_escaped;
}

void StringTestContext::InsertedWordCallback(const std::vector<char> &word) {
    // Nothing
}

thrust::host_vector<void *> StringTestContext::OutputBuffers() {
    return thrust::host_vector<void *>();
}

void StringTestContext::OutputValidate() {
    // Nothing
}


