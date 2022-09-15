#include "repeat_test_context.cuh"

void RepeatTestContext::OutputValidate() {
    // Nothing to validate
}

void RepeatTestContext::Initialize() {
    if (m_jsons.empty()) throw std::runtime_error("RepeatTestContext needs to be initialized with at least 1 json");
    const auto& longest = *std::max_element(
        m_jsons.begin(), m_jsons.end(), [](const std::string& a, const std::string& b) {
            return a.length() < b.length();
        });
    size_t max_len = longest.size();
    std::uniform_int_distribution<uint32_t> dist(0, m_jsons.size() - 1);
    std::vector<uint32_t> order(m_test_size, 0);
    std::iota(order.begin(), order.begin() + (long)m_jsons.size(), 0);
    std::generate(order.begin() + (long)m_jsons.size(), order.end(), [&]() { return dist(m_rand); });
    std::shuffle(order.begin(), order.end(), m_rand);
    m_h_input = thrust::host_vector<char>(m_test_size * max_len + 1);
    m_h_indices = thrust::host_vector<InputIndex>(m_test_size + 1);
    auto inp_it = m_h_input.data();
    auto ind_it = m_h_indices.begin();
    *ind_it = 0;
    ++ind_it;
    for (size_t i = 0; i < m_test_size; ++i)
    {
        const auto& str = m_jsons[order[i]];
        inp_it += snprintf(inp_it, max_len + 1, "%s", str.c_str());
        *ind_it = (inp_it - m_h_input.data());
        ++ind_it;
    }
    m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
    thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
    m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
}

thrust::host_vector<void *> RepeatTestContext::OutputBuffers() {
    return thrust::host_vector<void *>();
}

RepeatTestContext::RepeatTestContext(size_t test_size, size_t group_size, TestContext::SeedType seed) : TestContext(
    test_size, group_size, seed) {
}

void RepeatTestContext::AppendEntry(const std::string& json, bool valid) {
    m_jsons.push_back(json);
    m_valid.push_back(valid);
}

void RepeatTestContext::AppendEntry(std::string&& json, bool valid) {
    m_jsons.push_back(json);
    m_valid.push_back(valid);
}
