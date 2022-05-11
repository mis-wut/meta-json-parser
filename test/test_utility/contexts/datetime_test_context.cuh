#ifndef META_JSON_PARSER_DATETIME_TEST_CONTEXT_CUH
#define META_JSON_PARSER_DATETIME_TEST_CONTEXT_CUH
#include <clocale>
#include <cstdlib>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <type_traits>
#include "test_context.cuh"
#include <meta_json_parser/action/datetime/jdatetime.cuh>

template<class TimestampTypeT, class OutT>
class DatetimeTestContext : public TestContext {
    static_assert(
        std::is_same_v<TimestampTypeT, JDatetimeOptions::TimestampResolution::Seconds> ||
        std::is_same_v<TimestampTypeT, JDatetimeOptions::TimestampResolution::Milliseconds>,
        "TimestampTypeT must be either Seconds or Milliseconds."
    );
protected:
    using TimestampType = TimestampTypeT;
    using Out = OutT;

    thrust::host_vector<Out> m_h_correct;
    thrust::device_vector<Out> m_d_correct;
    thrust::device_vector<Out> m_d_result;
    std::string m_format;

    using Distribution = std::uniform_int_distribution<Out>;

    void OutputValidate() override {
        testing::AssertionResult result = testing::AssertionSuccess();
        if (!thrust::equal(m_d_correct.begin(), m_d_correct.end(), m_d_result.begin())) {
            thrust::host_vector<Out> h_result(m_d_result);
            auto mismatch = thrust::mismatch(m_h_correct.begin(), m_h_correct.end(), h_result.begin());
            size_t input_id = mismatch.first - m_h_correct.begin();
            size_t print_len = m_h_indices[input_id + 1] - m_h_indices[input_id];
            result = testing::AssertionFailure()
                    << "Mismatch output at " << input_id << " input value. "
                    << "Expected number \"" << *mismatch.first << "\", "
                    << "result number \"" << *mismatch.second << "\". "
                    << "Input was \"" << std::string_view(m_h_input.data() + m_h_indices[input_id], print_len) << "\".";
        }
        ASSERT_TRUE(result);
    }
public:
    DatetimeTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : TestContext(testSize, groupSize, seed),
              m_format("%Y-%m-%d") { }

    void Initialize() override {
        size_t max_len = 30;

        Distribution timestamp_dist(0, 4102444799); // 2099-12-31 23:59:59

        m_h_input = thrust::host_vector<char>(m_test_size * max_len + 1);
        m_h_indices = thrust::host_vector<InputIndex>(m_test_size + 1);
        m_h_correct = thrust::host_vector<Out>(m_test_size);
        m_d_result = thrust::device_vector<Out>(m_test_size, 0);

        auto inp_it = m_h_input.data();
        auto ind_it = m_h_indices.begin();
        *ind_it = 0;
        ++ind_it;
        std::string str(max_len, '\0');
        for (size_t i = 0; i < m_test_size; ++i)
        {
            std::time_t timestamp = timestamp_dist(m_rand);
            size_t to_print = std::strftime(str.data(), max_len, m_format.data(), std::gmtime(&timestamp));
            std::tm datetime{0};
            std::istringstream sinp(str);
            sinp >> std::get_time(&datetime, m_format.data());
            timestamp = std::mktime(&datetime) - timezone;
            // Check to faster detect issues with generation
            if (to_print > max_len) {
                FAIL() << '"' << str << "\" longer than max len " << max_len << "!\n";;
            }
            inp_it += snprintf(inp_it, to_print + 3, "\"%s\"", str.c_str());
            if constexpr (std::is_same_v<TimestampType, JDatetimeOptions::TimestampResolution::Milliseconds>) {
                m_h_correct[i] = static_cast<Out>(timestamp) * 1000;
            } else {
                m_h_correct[i] = static_cast<Out>(timestamp);
            }
            *ind_it = (inp_it - m_h_input.data());
            ++ind_it;
        }
        m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
        thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
        m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
        m_d_correct = thrust::device_vector<Out>(m_h_correct);
    }

#define Property(NAME, VAR)           \
auto Get##NAME() const {              \
    return VAR;                       \
}                                     \
void Set##NAME(decltype(VAR) value) { \
    VAR = value;                      \
}

    Property(DateFormat, m_format)

#undef Property

    thrust::host_vector<void *> OutputBuffers() override {
        thrust::host_vector<void *> result(1);
        result[0] = m_d_result.data().get();
        return result;
    }
};


#endif //META_JSON_PARSER_DATETIME_TEST_CONTEXT_CUH
