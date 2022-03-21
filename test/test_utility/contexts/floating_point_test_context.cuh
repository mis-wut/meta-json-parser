#ifndef META_JSON_PARSER_FLOATING_POINT_TEST_CONTEXT_CUH
#define META_JSON_PARSER_FLOATING_POINT_TEST_CONTEXT_CUH
#include <cstdlib>
#include <string>
#include "test_context.cuh"

template<class NumberT, class EpsilonT = std::ratio<1, 100000>>
struct AbsoluteDifferenceComparator : public thrust::binary_function<NumberT, NumberT, bool> {
    __host__ __device__ inline bool operator()(NumberT a, NumberT b) {
        static_assert(std::is_floating_point_v<NumberT>, "AbsoluteDifferenceComparator must operate on floating point types.");
        return false;
    }
};

template<class EpsilonT>
struct AbsoluteDifferenceComparator<float, EpsilonT> : public thrust::binary_function<float, float, bool> {
    __host__ __device__ inline bool operator()(float a, float b) {
        constexpr float epsilon = static_cast<float>(EpsilonT::num) / static_cast<float>(EpsilonT::den);
        if (a == b) {
            return true;
        }
        return fabsf(b - a) <= epsilon;
    }
};

template<class EpsilonT>
struct AbsoluteDifferenceComparator<double, EpsilonT> : public thrust::binary_function<double, double, bool> {
    __host__ __device__ inline bool operator()(double a, double b) {
        constexpr double epsilon = static_cast<double>(EpsilonT::num) / static_cast<double>(EpsilonT::den);
        if (a == b) {
            return true;
        }
        return fabs(b - a) <= epsilon;
    }
};

template<class NumberT, class EpsilonT = std::ratio<1, 100000>>
struct RelativeDifferenceComparator : public thrust::binary_function<NumberT, NumberT, bool> {
    __host__ __device__ inline bool operator()(NumberT a, NumberT b) {
        static_assert(std::is_floating_point_v<NumberT>, "AbsoluteDifferenceComparator must operate on floating point types.");
        return false;
    }
};

template<class EpsilonT>
struct RelativeDifferenceComparator<float, EpsilonT> : public thrust::binary_function<float, float, bool> {
    __host__ __device__ inline bool operator()(float a, float b) {
        constexpr float epsilon = static_cast<float>(EpsilonT::num) / static_cast<float>(EpsilonT::den);
        if (a == b) {
            return true;
        }
        const float diff = fabsf(a - b);
        if (a == 0 || b == 0) {
            return diff < std::numeric_limits<float>::min();
        }
        if (diff < std::numeric_limits<float>::min()) {
            return true;
        }
        return diff / fminf(fabsf(a) + fabsf(b), std::numeric_limits<float>::max()) < epsilon;
    }
};

template<class EpsilonT>
struct RelativeDifferenceComparator<double, EpsilonT> : public thrust::binary_function<double, double, bool> {
    __host__ __device__ inline bool operator()(double a, double b) {
        constexpr double epsilon = static_cast<double>(EpsilonT::num) / static_cast<double>(EpsilonT::den);
        if (a == b) {
            return true;
        }
        const double diff = fabs(a - b);
        if (a == 0 || b == 0) {
            return diff < std::numeric_limits<double>::min();
        }
        if (diff < std::numeric_limits<double>::min()) {
            return true;
        }
        return diff / fmin(fabs(a) + fabs(b), std::numeric_limits<double>::max()) < epsilon;
    }
};

struct GenerateOption {
    enum struct Options {
        Never,
        Sometimes,
        Always
    } option;

    GenerateOption(GenerateOption::Options opt) : option(opt) { }

    bool Use(std::minstd_rand& rand) const {
        static std::uniform_int_distribution<> gen(0, 1);
        switch (option) {
            case Options::Never:
                return false;
            case Options::Sometimes:
                return gen(rand);
            case Options::Always:
                return true;
            default:
                return false;
        }
    }

    bool Possible() const {
        switch (option) {
            case Options::Never:
                return false;
            case Options::Sometimes:
            case Options::Always:
                return true;
            default:
                return false;
        }
    }
};

template<class NumberT>
struct MantissaExpSquareDifference : public thrust::binary_function<NumberT, NumberT, thrust::pair<NumberT, NumberT>> {
    __host__ __device__ inline thrust::pair<NumberT, NumberT> operator()(NumberT a, NumberT b) {
        // infinity check
        if (a == b)
            return thrust::pair<NumberT, NumberT>(0, 0);
        int ae, be;
        NumberT am = template_cuda_math::frexp(a, &ae);
        NumberT bm = template_cuda_math::frexp(b, &be);
        return thrust::pair<NumberT, NumberT>(
            (am - bm) * (am - bm),
            (ae - be) * (ae - be)
        );
    }
};

template<class NumberT>
struct PairWiseSum : public thrust::binary_function<thrust::pair<NumberT, NumberT>, thrust::pair<NumberT, NumberT>, thrust::pair<NumberT, NumberT>> {
    __host__ __device__ inline thrust::pair<NumberT, NumberT> operator()(thrust::pair<NumberT, NumberT> a, thrust::pair<NumberT, NumberT> b) {
        return thrust::pair<NumberT, NumberT>(
            a.first + b.first,
            a.second + b.second
        );
    }
};

template<class NumberT, class ComparatorT>
class FloatingPointTestContext : public TestContext {
    static_assert(
            std::is_floating_point<NumberT>::value, "NumberT must be floating point type."
    );
protected:
    thrust::host_vector<NumberT> m_h_correct;
    thrust::device_vector<NumberT> m_d_correct;
    thrust::device_vector<NumberT> m_d_result;
    uint32_t m_min_digits;
    uint32_t m_max_digits;
    uint32_t m_min_fraction;
    uint32_t m_max_fraction;
    uint32_t m_min_exp;
    uint32_t m_max_exp;
    GenerateOption m_use_exp;
    GenerateOption m_use_exp_sign;
    GenerateOption m_use_fraction;
    GenerateOption m_use_sign;

    using Distribution = std::uniform_int_distribution<uint32_t>;
    using DistributionChar = std::uniform_int_distribution<char>;

    using Comparator = ComparatorT;

    virtual void InsertedNumberCallback(size_t index, NumberT value) {
        m_h_correct[index] = value;
    }

    void GetMeanSquareError(double& mantissa, double& exp) {
        thrust::device_vector<thrust::pair<NumberT, NumberT>> sq_diff(m_d_result.size());
        thrust::transform(
            m_d_correct.begin(), m_d_correct.end(), m_d_result.begin(), sq_diff.begin(), MantissaExpSquareDifference<NumberT>()
        );
        auto sum = thrust::reduce(sq_diff.begin(), sq_diff.end(), thrust::pair<NumberT, NumberT>(0.0, 0.0), PairWiseSum<NumberT>());
        mantissa = static_cast<double>(sum.first) / m_d_result.size();
        exp = static_cast<double>(sum.second) / m_d_result.size();
    }

    void OutputValidate() override {
        Comparator comparator;
        testing::AssertionResult result = testing::AssertionSuccess();
        if (!thrust::equal(m_d_correct.begin(), m_d_correct.end(), m_d_result.begin(), comparator)) {
            thrust::host_vector<NumberT> h_result(m_d_result);
            auto mismatch = thrust::mismatch(m_h_correct.begin(), m_h_correct.end(), h_result.begin(), comparator);
            size_t input_id = mismatch.first - m_h_correct.begin();
            size_t print_len = m_h_indices[input_id + 1] - m_h_indices[input_id];
            result = testing::AssertionFailure()
                    << "Mismatch output at " << input_id << " input value. "
                    << "Expected number \"" << static_cast<NumberT>(*mismatch.first) << "\", "
                    << "result number \"" << static_cast<NumberT>(*mismatch.second) << "\". "
                    << "Input was \"" << std::string_view(m_h_input.data() + m_h_indices[input_id], print_len) << "\".";
        } else {
            double mantissa, exp;
            GetMeanSquareError(mantissa, exp);
            std::cout << "Mantissa mean square error: " << std::setprecision(10) << mantissa << ".\n";
            std::cout << "Exponent mean square error: " << std::setprecision(10) << exp << ".\n";
        }
        ASSERT_TRUE(result);
    }
public:
    FloatingPointTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : TestContext(testSize, groupSize, seed),
              m_min_digits(0), m_max_digits(5),
              m_min_fraction(1), m_max_fraction(5),
              m_min_exp(1), m_max_exp(2),
              m_use_sign(GenerateOption::Options::Sometimes),
              m_use_fraction(GenerateOption::Options::Sometimes),
              m_use_exp(GenerateOption::Options::Sometimes), m_use_exp_sign(GenerateOption::Options::Sometimes) { }

    void Initialize() override {
        size_t max_len =
            (m_use_sign.Possible() ? 1 : 0) +
            m_max_digits +
            (m_use_fraction.Possible() ? 1 + m_max_fraction  : 0) +
            (m_use_exp.Possible()
                ? 1 + m_max_exp + (m_use_exp_sign.Possible() ? 1 : 0)
                : 0
            );

        DistributionChar start_digit('1', '9');
        DistributionChar digit('0', '9');
        Distribution bool_dist(0, 1);

        Distribution digit_len(m_min_digits, m_max_digits);
        Distribution fraction_len(m_min_fraction, m_max_fraction);
        Distribution exp_len(m_min_exp, m_max_exp);

        m_h_input = thrust::host_vector<char>(m_test_size * max_len + 1);
        m_h_indices = thrust::host_vector<InputIndex>(m_test_size + 1);
        m_h_correct = thrust::host_vector<NumberT>(m_test_size);
        m_d_result = thrust::device_vector<NumberT>(m_test_size, 0);

        auto inp_it = m_h_input.data();
        auto ind_it = m_h_indices.begin();
        *ind_it = 0;
        ++ind_it;
        for (size_t i = 0; i < m_test_size; ++i)
        {
            std::stringstream stream;
            if (m_use_sign.Use(m_rand)) {
                stream << '-';
            }
            // len == 0 means 0 as digit number
            auto len = digit_len(m_rand);
            if (len == 0) {
                stream << '0';
            } else if (len == 1) {
                stream << digit(m_rand);
            } else {
                stream << start_digit(m_rand);
                for (int j = len - 1; j > 0; --j) {
                    stream << digit(m_rand);
                }
            }

            bool is_fraction = m_use_fraction.Use(m_rand);
            if (is_fraction) {
                stream << '.';
                len = fraction_len(m_rand);
                for (int j = len; j > 0; --j) {
                    stream << digit(m_rand);
                }
            }

            bool is_exp = m_use_exp.Use(m_rand);
            if (is_exp) {
                stream << (bool_dist(m_rand) ? 'e' : 'E');
                if (m_use_exp_sign.Use(m_rand)) {
                    if (bool_dist(m_rand)) {
                        stream << '+';
                    } else {
                        stream << '-';
                    }
                }

                len = exp_len(m_rand);
                for (int j = len; j > 0; --j) {
                    stream << digit(m_rand);
                }
            }

            auto str = stream.str();
            NumberT val;

            std::string parse_str;

            //stod/stof cannot parse value like 1e23 (string with exponent without fraction)
            if (!is_fraction && is_exp) {
                auto exp_off = str.find('e');
                if (exp_off == std::string::npos)
                    exp_off = str.find('E');
                auto digits = std::string(str.data(), exp_off);
                auto exponent = std::string(str.data() + exp_off + 1, str.data() + str.length());
                std::stringstream stream;
                stream << digits << ".0e" << exponent;
                parse_str = stream.str();
            } else {
                parse_str = str;
            }

            try {
                if constexpr (std::is_same_v<NumberT, double>) {
                    val = static_cast<NumberT>(std::stod(parse_str));
                } else {
                    val = static_cast<NumberT>(std::stof(parse_str));
                }
            } catch (std::invalid_argument& e){
                FAIL() << "stof reported \"" << str << "\" as invalid argument.";
            } catch (std::out_of_range& e){
                if (str.find("e-") != std::string::npos || str.find("E-") != std::string::npos) {
                    val = static_cast<NumberT>(0);
                } else {
                    val = std::numeric_limits<NumberT>::infinity();
                }
                if (str[0] == '-')
                    val = -val;
            }

            // Check to faster detect issues with generation
            if (str.length() > max_len) {
                FAIL() << '"' << str << "\" longer than max len " << max_len << "!\n";;
            }
            inp_it += snprintf(inp_it, max_len + 1, "%s", str.c_str());
            InsertedNumberCallback(i, val);
            *ind_it = (inp_it - m_h_input.data());
            ++ind_it;
        }
        m_d_input = thrust::device_vector<char>(m_h_input.size() + 256); //256 to allow batch loading
        thrust::copy(m_h_input.begin(), m_h_input.end(), m_d_input.begin());
        m_d_indices = thrust::device_vector<InputIndex>(m_h_indices);
        m_d_correct = thrust::device_vector<NumberT>(m_h_correct);
    }

#define Property(NAME, VAR)           \
auto Get##NAME() const {              \
    return VAR;                       \
}                                     \
void Set##NAME(decltype(VAR) value) { \
    VAR = value;                      \
}

    Property(MinDigits, m_min_digits)
    Property(MaxDigits, m_max_digits)
    Property(MinFraction, m_min_fraction)
    Property(MaxFraction, m_max_fraction)
    Property(MinExp, m_min_exp)
    Property(MaxExp, m_max_exp)
    Property(UseSign, m_use_sign)
    Property(UseFraction, m_use_fraction)
    Property(UseExp, m_use_exp)
    Property(UseExpSign, m_use_exp_sign)

#undef Property

    thrust::host_vector<void *> OutputBuffers() override {
        thrust::host_vector<void *> result(1);
        result[0] = m_d_result.data().get();
        return result;
    }
};

#endif //META_JSON_PARSER_FLOATING_POINT_TEST_CONTEXT_CUH
