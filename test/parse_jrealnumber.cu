#include <gtest/gtest.h>
#include <meta_json_parser/action/jrealnumber.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/floating_point_test_context.cuh"
#include "test_utility/test_launcher.cuh"

class ParseJRealNumberTest : public ::testing::Test { };

using Opts = GenerateOption::Options;

template<class OutTypeT, int GroupSizeT>
void templated_ParseFloatInteger()
{
    using Epsilon = std::ratio<1, 100'000>;
    using Comparator = RelativeDifferenceComparator<OutTypeT, Epsilon>;
    FloatingPointTestContext<OutTypeT, Comparator> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetUseSign(Opts::Sometimes);
    context.SetUseFraction(Opts::Never);
    context.SetUseExp(Opts::Never);
    context.SetMinDigits(0);
    context.SetMaxDigits(static_cast<uint32_t>(floor(log10(std::numeric_limits<OutTypeT>::max()))));
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<
            JRealNumberOptions::JRealNumberExponent,
            JRealNumberOptions::JRealNumberExponent::WithoutExponent
        >
    >;
    using BA = JRealNumber<OutTypeT, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

template<class OutTypeT, int GroupSizeT>
void templated_ParseFloatFraction() {
    using Epsilon = std::ratio<1, 100'000>;
    using Comparator = RelativeDifferenceComparator<OutTypeT, Epsilon>;
    FloatingPointTestContext<OutTypeT, Comparator> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetUseSign(Opts::Sometimes);
    context.SetUseFraction(Opts::Always);
    context.SetUseExp(Opts::Never);
    context.SetMinDigits(0);
    context.SetMaxDigits(static_cast<uint32_t>(floor(log10(std::numeric_limits<OutTypeT>::max()))));
    context.SetMinFraction(1);
    context.SetMaxFraction(20);
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<
            JRealNumberOptions::JRealNumberExponent,
            JRealNumberOptions::JRealNumberExponent::WithoutExponent
        >
    >;
    using BA = JRealNumber<OutTypeT, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

template<class OutTypeT, int GroupSizeT>
void templated_ParseFloatExponent() {
    using Epsilon = std::ratio<1, 100'000>;
    using Comparator = RelativeDifferenceComparator<OutTypeT, Epsilon>;
    FloatingPointTestContext<OutTypeT, Comparator> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetUseSign(Opts::Sometimes);
    context.SetUseFraction(Opts::Never);
    context.SetUseExp(Opts::Always);
    context.SetUseExpSign(Opts::Sometimes);
    context.SetMinDigits(0);
    context.SetMaxDigits(static_cast<uint32_t>(floor(log10(std::numeric_limits<OutTypeT>::max()))));
    context.SetMinExp(1);
    context.SetMaxExp(std::is_same_v<OutTypeT, double> ? 3 : 2);
    context.Initialize();
    using BA = JRealNumber<OutTypeT, void>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

template<class OutTypeT, int GroupSizeT>
void templated_ParseFloatAll() {
    using Epsilon = std::ratio<1, 100'000>;
    using Comparator = RelativeDifferenceComparator<OutTypeT, Epsilon>;
    FloatingPointTestContext<OutTypeT, Comparator> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetUseSign(Opts::Sometimes);
    context.SetUseFraction(Opts::Sometimes);
    context.SetUseExp(Opts::Sometimes);
    context.SetUseExpSign(Opts::Sometimes);
    context.SetMinDigits(0);
    context.SetMaxDigits(static_cast<uint32_t>(floor(log10(std::numeric_limits<OutTypeT>::max()))));
    context.SetMinFraction(1);
    context.SetMaxFraction(20);
    context.SetMinExp(1);
    context.SetMaxExp(std::is_same_v<OutTypeT, double> ? 3 : 2);
    context.Initialize();
    using BA = JRealNumber<OutTypeT, void>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_FloatDouble(mfun) \
META_WS_4(mfun, float)\
META_WS_4(mfun, double)

#define META_ParseJRealNumberTest(WS, TYPE)\
TEST_F(ParseJRealNumberTest, Integer_##TYPE##_W##WS) {\
	templated_ParseFloatInteger<TYPE, WS>();\
}\
TEST_F(ParseJRealNumberTest, Fraction_##TYPE##_W##WS) {\
	templated_ParseFloatFraction<TYPE, WS>();\
}\
TEST_F(ParseJRealNumberTest, Exponent_##TYPE##_W##WS) {\
	templated_ParseFloatExponent<TYPE, WS>();\
}\
TEST_F(ParseJRealNumberTest, All_##TYPE##_W##WS) {\
	templated_ParseFloatAll<TYPE, WS>();\
}

META_FloatDouble(META_ParseJRealNumberTest)
