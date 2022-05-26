#include <gtest/gtest.h>
#include <thrust/logical.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>
#include <boost/mp11.hpp>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jarray.cuh>
#include "test_utility/contexts/test_context.cuh"
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"

using namespace boost::mp11;

template<class T>
class ParseJArray2UIntSuite : public ::testing::Test {};

template<class T>
class ParseJArrayWorkgroupSuite : public ::testing::Test {};

template<class OutType1T, class OutType2T>
struct TestContextArray2UInt : public TestContext {
	thrust::host_vector<OutType1T> m_h_correct_1;
	thrust::host_vector<OutType2T> m_h_correct_2;
	thrust::device_vector<OutType1T> m_d_correct_1;
    thrust::device_vector<OutType2T> m_d_correct_2;
    thrust::device_vector<OutType1T> m_d_result_1;
	thrust::device_vector<OutType2T> m_d_result_2;
	std::string m_format;

	bool m_check_column_1;
    bool m_check_column_2;

public:
    TestContextArray2UInt(std::string&& format, size_t test_size, size_t group_size, SeedType seed)
    : TestContext(test_size, group_size, seed), m_format(format), m_check_column_1(true), m_check_column_2(true) {}

    void Initialize() override {
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
        size_t MAX_LEN = MAX_UINT_LEN_1 + MAX_UINT_LEN_2 + m_format.length() - 8; // Assumption that there are 2 %llu in format
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
            auto x1 = m_h_correct_1[i];
            auto x2 = m_h_correct_2[i];
            inp_it += snprintf(inp_it, MAX_LEN + 1, m_format.c_str(), static_cast<long long unsigned int>(x1), static_cast<long long unsigned int>(x2));
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

    thrust::host_vector<void *> OutputBuffers() override {
        thrust::host_vector<void *> result;
        if (m_check_column_1)
            result.push_back(reinterpret_cast<void*>(m_d_result_1.data().get()));
        if (m_check_column_2)
            result.push_back(reinterpret_cast<void*>(m_d_result_2.data().get()));
        return result;
    }

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
            thrust::host_vector<OutType> h_correct, std::string&& name){
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

    void OutputValidate() override {
        if (m_check_column_1)
            UIntValidate(m_d_result_1, m_d_correct_1, m_h_correct_1, "first column");
        if (m_check_column_2)
            UIntValidate(m_d_result_2, m_d_correct_2, m_h_correct_2, "second column");
    }
};

TYPED_TEST_SUITE_P(ParseJArray2UIntSuite);
TYPED_TEST_SUITE_P(ParseJArrayWorkgroupSuite);

TYPED_TEST_P(ParseJArray2UIntSuite, ParseArrayWith2Uints) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using OutTypes = mp_at_c<TypeParam, 1>;
    using OutType1 = mp_at_c<OutTypes, 0>;
    using OutType2 = mp_at_c<OutTypes, 1>;
    using Skipping = mp_at_c<TypeParam, 2>;

    using Zero = mp_int<0>;
    using One = mp_int<1>;
    using BA = JArray<ArrayEntries<
        Zero, JNumber<OutType1, Zero>,
        One, JNumber<OutType2, One>
    >, mp_list<
        mp_list<
            JArrayOptions::Skip,
            Skipping
        >
    >>;

    TestContextArray2UInt<OutType1, OutType2> context("[ %llu, %llu ]", TEST_SIZE, WorkGroupSize::value, SEED);
    context.Initialize();
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(ParseJArray2UIntSuite, ParseNestedArrays) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using OutTypes = mp_at_c<TypeParam, 1>;
    using OutType1 = mp_at_c<OutTypes, 0>;
    using OutType2 = mp_at_c<OutTypes, 1>;
    using Skipping = mp_at_c<TypeParam, 2>;

    using Zero = mp_int<0>;
    using One = mp_int<1>;
    using BA = JArray<ArrayEntries<
        Zero, JArray<ArrayEntries<
            Zero, JNumber<OutType1, Zero>
        >>,
        One, JArray<ArrayEntries<
            Zero, JArray<ArrayEntries<
                Zero, JNumber<OutType2, One>
            >>
        >>
    >, mp_list<
        mp_list<
            JArrayOptions::Skip,
            Skipping
        >
    >>;

    TestContextArray2UInt<OutType1, OutType2> context("[ [ %llu],[[%llu] ]]", TEST_SIZE, WorkGroupSize::value, SEED);
    context.Initialize();
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(ParseJArrayWorkgroupSuite, Skipping) {
    using WorkGroupSize = TypeParam;

    using Two = mp_int<2>;
    using BA = JArray<ArrayEntries<
        Two, JNumber<uint32_t, Two>
    >, mp_list<
        mp_list<
            JArrayOptions::Skip,
            JArrayOptions::Skip::Enable_c<8>
        >
    >>;

    TestContextArray2UInt<uint32_t, uint64_t> context(
        R"JSON([ "skip me", { "key": [ 213, true], "next": 0.4e-4}, %llu, null, [ [[%llu ] ] ] ])JSON",
        TEST_SIZE, WorkGroupSize::value, SEED
    );
    context.SetColumn2Check(false);
    context.Initialize();
    LaunchTest<BA, WorkGroupSize>(context);
}

REGISTER_TYPED_TEST_SUITE_P(ParseJArray2UIntSuite, ParseArrayWith2Uints, ParseNestedArrays);
REGISTER_TYPED_TEST_SUITE_P(ParseJArrayWorkgroupSuite, Skipping);

using UnsignedTypes = mp_list<
    mp_list<uint8_t, uint64_t>,
    mp_list<uint32_t, uint16_t>,
    mp_list<uint16_t, uint8_t>
>;

using SkippingOption = mp_list<
    JArrayOptions::Skip::Disable,
    JArrayOptions::Skip::Enable_c<8>
>;

using AllWorkGroupsWith2UIntTypes = mp_rename<mp_product<
    mp_list,
    AllWorkGroups,
    UnsignedTypes,
    SkippingOption
>, ::testing::Types>;

using AllWorkGroupsTypes = mp_rename<AllWorkGroups, ::testing::Types>;

struct NameGenerator {
    template <typename TypeParam>
    static std::string GetName(int i) {
        using GroupSize = mp_at_c<TypeParam, 0>;
        using OutTypes = mp_at_c<TypeParam, 1>;
        using OutType1 = mp_at_c<OutTypes, 0>;
        using OutType2 = mp_at_c<OutTypes, 1>;
        using Skipping = mp_at_c<TypeParam, 2>;

        std::stringstream stream;
        stream << "WS_" << GroupSize::value;
        stream << "_";

        for (int j = 0; j < 2; ++j) {
            int size = j == 0 ? sizeof(OutType1) : sizeof (OutType2);
            switch (size) {
                case 1:
                    stream << "uint8";
                    break;
                case 2:
                    stream << "uint16";
                    break;
                case 4:
                    stream << "uint32";
                    break;
                case 8:
                    stream << "uint64";
                    break;
                default:
                    stream << "UNKNOWN";
                    break;
            }
            stream << "_";
        }

        if constexpr (std::is_same_v<Skipping, JArrayOptions::Skip::Disable>) {
            stream << "skip_disabled";
        } else {
            stream << "skip_enabled";
        }

        return stream.str();
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroupsWith2UInt, ParseJArray2UIntSuite, AllWorkGroupsWith2UIntTypes, NameGenerator);
INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroups, ParseJArrayWorkgroupSuite, AllWorkGroupsTypes, WorkGroupNameGenerator);

