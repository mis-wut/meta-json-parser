#ifndef META_JSON_PARSER_NUMBER_TEST_CONTEXT_CUH
#define META_JSON_PARSER_NUMBER_TEST_CONTEXT_CUH
#include <random>
#include <type_traits>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <boost/mp11/utility.hpp>
#include "test_context.cuh"

//To suppress warning about pointless comparison of unsigned integer with zero
template<class NumberT, std::enable_if_t<!std::is_unsigned<NumberT>::value, bool> = true>
constexpr bool IsLessThan0(NumberT val) { return val < 0; }
template<class NumberT, std::enable_if_t<std::is_unsigned<NumberT>::value, bool> = true>
constexpr bool IsLessThan0(NumberT) { return false; }


template<class NumberT>
struct IntegerComparator : public thrust::binary_function<NumberT, NumberT, bool> {
    __host__ __device__ inline bool operator()(NumberT a, NumberT b) { return a == b; }
};

template<class NumberT, class EpsilonT = std::ratio<1, 100000>>
struct FloatingComparator : public thrust::binary_function<NumberT, NumberT, bool> {
    __host__ __device__ inline bool operator()(NumberT a, NumberT b) {
        static_assert(std::is_floating_point_v<NumberT>, "FloatingComparator must operate on floating point types.");
        return false;
    }
};

template<class EpsilonT>
struct FloatingComparator<float, EpsilonT> : public thrust::binary_function<float, float, bool> {
    __host__ __device__ inline bool operator()(float a, float b) {
        constexpr float epsilon = static_cast<float>(EpsilonT::num) / static_cast<float>(EpsilonT::den);
        return fabsf(b - a) <= epsilon;
    }
};

template<class EpsilonT>
struct FloatingComparator<double, EpsilonT> : public thrust::binary_function<double, double, bool> {
    __host__ __device__ inline bool operator()(double a, double b) {
        constexpr double epsilon = static_cast<double>(EpsilonT::num) / static_cast<double>(EpsilonT::den);
        return fabs(b - a) <= epsilon;
    }
};

template<class CalculationT, class GenerateT = void>
class NumberTestContext : public TestContext {
    static_assert(
            std::is_integral<CalculationT>::value || std::is_floating_point<CalculationT>::value,
        "NumberT must be either integral or floating point type."
    );
protected:
    using Calculation = CalculationT;
    using Generate = boost::mp11::mp_if<
            std::is_same<GenerateT, void>,
            Calculation,
            GenerateT
    >;

    thrust::host_vector<Calculation> m_h_correct;
    thrust::device_vector<Calculation> m_d_correct;
    thrust::device_vector<Calculation> m_d_result;
    Generate m_min_val;
    Generate m_max_val;
    size_t m_max_precision;

    using RepresentType = std::common_type_t<
        Generate,
        boost::mp11::mp_if<
            std::is_unsigned<Generate>,
            uint64_t,
            int64_t
        >
    >;

    using IntegralDistribution = boost::mp11::mp_eval_if_not<
        std::is_integral<Generate>,
        void,
        std::uniform_int_distribution,
        Generate
    >;

    using RealDistribution = boost::mp11::mp_eval_if_not<
        std::is_floating_point<Generate>,
        void,
        std::uniform_real_distribution,
        Generate
    >;

    using Distribution = boost::mp11::mp_if<
        std::is_integral<Generate>,
        IntegralDistribution,
        RealDistribution
    >;

    using Comparator = boost::mp11::mp_eval_if_not<
        std::is_floating_point<Calculation>,
        IntegerComparator<Calculation>,
        FloatingComparator,
        Calculation,
        std::ratio<1, 100000>
    >;

    virtual void InsertedNumberCallback(size_t index, Calculation value) {
        m_h_correct[index] = value;
    }

    void OutputValidate() override {
        Comparator comparator;
        testing::AssertionResult result = testing::AssertionSuccess();
        if (!thrust::equal(m_d_correct.begin(), m_d_correct.end(), m_d_result.begin(), comparator)) {
            thrust::host_vector<Calculation> h_result(m_d_result);
            auto mismatch = thrust::mismatch(m_h_correct.begin(), m_h_correct.end(), h_result.begin(), comparator);
            size_t input_id = mismatch.first - m_h_correct.begin();
            size_t print_len = m_h_indices[input_id + 1] - m_h_indices[input_id];
            result = testing::AssertionFailure()
                    << "Mismatch output at " << input_id << " input value. "
                    << "Expected number \"" << static_cast<Calculation>(*mismatch.first) << "\", "
                    << "result number \"" << static_cast<Calculation>(*mismatch.second) << "\"."
                    << "Input was \"" << std::string_view(m_h_input.data() + m_h_indices[input_id], print_len);
        }
        ASSERT_TRUE(result);
    }
public:
    NumberTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : TestContext(testSize, groupSize, seed),
              m_min_val(std::numeric_limits<Generate>::min()),
              m_max_val(std::numeric_limits<Generate>::max()),
              m_max_precision(3) { }

    void Initialize() override {
        size_t max_len = static_cast<size_t>(std::ceil(std::max(
                std::log10(std::fabs(m_max_val)),
                std::log10(std::fabs(m_min_val))
        )));
        if (IsLessThan0(m_min_val))
            max_len += 1;
        if (std::is_floating_point<Generate>::value) {
            max_len += 1 + m_max_precision;
        }
        Distribution dist(m_min_val, m_max_val);
        m_h_input = thrust::host_vector<char>(m_test_size * max_len + 1);
        m_h_indices = thrust::host_vector<InputIndex>(m_test_size + 1);
        m_h_correct = thrust::host_vector<Calculation>(m_test_size);
        m_d_result = thrust::device_vector<Calculation>(m_test_size, 0);
        std::generate(m_h_correct.begin(), m_h_correct.end(), [&]() { return static_cast<Calculation>(dist(this->m_rand)); });
        auto inp_it = m_h_input.data();
        auto ind_it = m_h_indices.begin();
        *ind_it = 0;
        ++ind_it;
        for (size_t i = 0; i < m_test_size; ++i)
        {
            RepresentType val = m_h_correct[i];
            std::stringstream stream;
            stream << std::setprecision(m_max_precision) << val;
            auto str = stream.str();
            // Check to faster detect issues with generation
            if (str.length() > max_len) {
                std::cout << '"' << str << "\" longer than max len " << max_len << "!\n";
                FAIL();
            }
            inp_it += snprintf(inp_it, max_len + 1, "%s", str.c_str());
            InsertedNumberCallback(i, val);
            *ind_it = (inp_it - m_h_input.data());
            ++ind_it;
        }
        m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
        thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
        m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
        m_d_correct = thrust::device_vector<Calculation>(m_h_correct);
    }

    Generate GetMinimumValue() const {
        return m_min_val;
    }
    void SetMinimumValue(Generate min_val) {
        m_min_val = min_val;
    }

    Generate GetMaximumValue() const {
        return m_max_val;
    }
    void SetMaximumValue(Generate max_val) {
        m_max_val = max_val;
    }

    size_t GetMaximumPrecision() const {
        return m_max_precision;
    }
    void SetMaximumPrecision(size_t max_precision) {
        m_max_precision = max_precision;
    }

    thrust::host_vector<void *> OutputBuffers() override {
        thrust::host_vector<void *> result(1);
        result[0] = m_d_result.data().get();
        return result;
    }
};


#endif //META_JSON_PARSER_NUMBER_TEST_CONTEXT_CUH
