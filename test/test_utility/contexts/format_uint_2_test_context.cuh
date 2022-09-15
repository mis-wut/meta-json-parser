#ifndef META_JSON_PARSER_FORMAT2UINT_CUH
#define META_JSON_PARSER_FORMAT2UINT_CUH
#include <utility>
#include <initializer_list>
#include <string>
#include <gtest/gtest.h>
#include <boost/mp11.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "test_context.cuh"

struct Uint2Format {
    std::string m_format;
    enum class Order {
        FirstSecond,
        SecondFirst
    } m_order;

    Uint2Format(std::string& format, Order order = Order::FirstSecond) : m_format(format), m_order(order) { }
    Uint2Format(std::string&& format, Order order = Order::FirstSecond) : m_format(format), m_order(order) { }

    Uint2Format(const Uint2Format&) = default;
    Uint2Format& operator=(const Uint2Format&) = default;
    Uint2Format(Uint2Format&&) = default;
    Uint2Format& operator=(Uint2Format&&) = default;

    ~Uint2Format() = default;
};

template<class OutType1T, class OutType2T>
struct FormatUint2TestContext : public TestContext {
    thrust::host_vector<OutType1T> m_h_correct_1;
    thrust::host_vector<OutType2T> m_h_correct_2;
    thrust::device_vector<OutType1T> m_d_correct_1;
    thrust::device_vector<OutType2T> m_d_correct_2;
    thrust::device_vector<OutType1T> m_d_result_1;
    thrust::device_vector<OutType2T> m_d_result_2;
    std::vector<Uint2Format> m_formats;

    bool m_check_column_1;
    bool m_check_column_2;

public:
    FormatUint2TestContext(std::string&& format, size_t test_size, size_t group_size, SeedType seed)
        : TestContext(test_size, group_size, seed), m_formats({format}), m_check_column_1(true), m_check_column_2(true) {}

    FormatUint2TestContext(const std::vector<Uint2Format>& formats, size_t test_size, size_t group_size, SeedType seed)
        : TestContext(test_size, group_size, seed), m_formats(formats), m_check_column_1(true), m_check_column_2(true) {}

    FormatUint2TestContext(std::vector<Uint2Format>&& formats, size_t test_size, size_t group_size, SeedType seed)
        : TestContext(test_size, group_size, seed), m_formats(formats), m_check_column_1(true), m_check_column_2(true) {}

    FormatUint2TestContext(std::initializer_list<Uint2Format> formats, size_t test_size, size_t group_size, SeedType seed)
        : TestContext(test_size, group_size, seed), m_formats(formats.begin(), formats.end()), m_check_column_1(true), m_check_column_2(true) {}

    void Initialize() override;

    thrust::host_vector<void *> OutputBuffers() override;

    void SetColumn1Check(bool value) {
        m_check_column_1 = value;
    }

    void SetColumn2Check(bool value) {
        m_check_column_2 = value;
    }

protected:
    template<class OutType>
    void UIntValidate(
        thrust::device_vector<OutType> d_result, thrust::device_vector<OutType> d_correct,
        thrust::host_vector<OutType> h_correct, std::string&& name);

    void OutputValidate() override {
        if (m_check_column_1)
            UIntValidate(m_d_result_1, m_d_correct_1, m_h_correct_1, "first column");
        if (m_check_column_2)
            UIntValidate(m_d_result_2, m_d_correct_2, m_h_correct_2, "second column");
    }
};

template<class OutType1T, class OutType2T>
void FormatUint2TestContext<OutType1T, OutType2T>::Initialize() {
    using Generate1T = boost::mp11::mp_if_c<sizeof(OutType1T) == 1, uint16_t, OutType1T>;
    using Generate2T = boost::mp11::mp_if_c<sizeof(OutType1T) == 1, uint16_t, OutType1T>;
    Generate1T MAX_VAL_1 = static_cast<Generate1T>(std::numeric_limits<OutType1T>::max() - 1);
    Generate2T MAX_VAL_2 = static_cast<Generate2T>(std::numeric_limits<OutType2T>::max() - 1);
    size_t MAX_UINT_LEN_1 = (size_t)std::ceil(std::log10((double)MAX_VAL_1));
    size_t MAX_UINT_LEN_2 = (size_t)std::ceil(std::log10((double)MAX_VAL_2));
    if (MAX_UINT_LEN_1 > m_group_size - 1)
    {
        MAX_VAL_1 = 1;
        for (int i = 0; i < m_group_size - 1; ++i)
            MAX_VAL_1 *= 10;
        MAX_VAL_1 -= 1;
        MAX_UINT_LEN_1 = m_group_size - 1;
    }
    if (MAX_UINT_LEN_2 > m_group_size - 1)
    {
        MAX_VAL_2 = 1;
        for (int i = 0; i < m_group_size - 1; ++i)
            MAX_VAL_2 *= 10;
        MAX_VAL_2 -= 1;
        MAX_UINT_LEN_2 = m_group_size - 1;
    }
    std::minstd_rand rng;
    std::uniform_int_distribution<Generate1T> dist_1(1, MAX_VAL_1);
    std::uniform_int_distribution<Generate2T> dist_2(1, MAX_VAL_2);
    std::uniform_int_distribution<uint32_t> format_dist(0, m_formats.size() - 1);
    size_t MAX_FORMAT_LEN = 0;
    for (const auto& format : m_formats) {
        MAX_FORMAT_LEN = std::max(format.m_format.length(), MAX_FORMAT_LEN);
    }
    size_t MAX_LEN = MAX_UINT_LEN_1 + MAX_UINT_LEN_2 + MAX_FORMAT_LEN - 8; // Assumption that there are 2 %llu in format
    m_h_input = thrust::host_vector<char>(m_test_size * MAX_LEN);
    m_h_correct_1 = thrust::host_vector<OutType1T>(m_test_size);
    m_h_correct_2 = thrust::host_vector<OutType2T>(m_test_size);
    m_h_indices = thrust::host_vector<InputIndex>(m_test_size + 1);
    std::generate(m_h_correct_1.begin(), m_h_correct_1.end(), [&dist_1, &rng]() { return static_cast<OutType1T>(dist_1(rng)); });
    std::generate(m_h_correct_2.begin(), m_h_correct_2.end(), [&dist_2, &rng]() { return static_cast<OutType2T>(dist_2(rng)); });
    auto inp_it = m_h_input.data();
    auto ind_it = m_h_indices.begin();
    *ind_it = 0;
    ++ind_it;
    for (size_t i = 0; i < m_test_size; ++i)
    {
        const auto& format = m_formats[format_dist(m_rand)];
        uint64_t x1 = static_cast<uint64_t>(m_h_correct_1[i]);
        uint64_t x2 = static_cast<uint64_t>(m_h_correct_2[i]);
        if (format.m_order == Uint2Format::Order::SecondFirst)
            std::swap(x1, x2);
        inp_it += snprintf(inp_it, MAX_LEN + 1, format.m_format.c_str(), static_cast<long long unsigned int>(x1), static_cast<long long unsigned int>(x2));
        *ind_it = (inp_it - m_h_input.data());
        ++ind_it;
    }
    m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
    thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
    m_d_correct_1 = thrust::device_vector<OutType1T>(m_h_correct_1);
    m_d_correct_2 = thrust::device_vector<OutType2T>(m_h_correct_2);
    m_d_result_1 = thrust::device_vector<OutType1T>(m_h_correct_1.size());
    m_d_result_2 = thrust::device_vector<OutType2T>(m_h_correct_2.size());
    m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
}

template<class OutType1T, class OutType2T>
thrust::host_vector<void *> FormatUint2TestContext<OutType1T, OutType2T>::OutputBuffers() {
    thrust::host_vector<void *> result;
    if (m_check_column_1)
        result.push_back(reinterpret_cast<void*>(m_d_result_1.data().get()));
    if (m_check_column_2)
        result.push_back(reinterpret_cast<void*>(m_d_result_2.data().get()));
    return result;
}

template<class OutType1T, class OutType2T>
template<class OutType>
void FormatUint2TestContext<OutType1T, OutType2T>::UIntValidate(thrust::device_vector<OutType> d_result,
                                                                thrust::device_vector<OutType> d_correct,
                                                                thrust::host_vector<OutType> h_correct,
                                                                std::string &&name) {
    testing::AssertionResult assertion_result = testing::AssertionSuccess();
    if (!thrust::equal(d_correct.begin(), d_correct.end(), d_result.begin())) {
        thrust::host_vector<OutType1T> h_result(d_result);
        auto mismatch = thrust::mismatch(h_correct.begin(), h_correct.end(), h_result.begin());
        size_t input_id = mismatch.first - h_correct.begin();
        size_t print_len = m_h_indices[input_id + 1] - m_h_indices[input_id];
        assertion_result = testing::AssertionFailure()
            << "Mismatch output at " << input_id << " input value in " << name << ". "
            << "Expected number \"" << static_cast<OutType>(*mismatch.first) << "\", "
            << "result number \"" << static_cast<OutType>(*mismatch.second) << "\"."
            << "Input was \"" << std::string_view(m_h_input.data() + m_h_indices[input_id], print_len);
    }
    ASSERT_TRUE(assertion_result);
}

#endif //META_JSON_PARAT2UINT_CUH
