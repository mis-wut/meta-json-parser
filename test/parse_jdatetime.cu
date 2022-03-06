#include <string>
#include <gtest/gtest.h>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"
#include "test_utility/contexts/datetime_test_context.cuh"
#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/mp_string.h>

class ParseJDataTest : public ::testing::Test { };

using namespace JsonParsers::DatetimeTokens;
using namespace boost::mp11;

template<class TimestampResolutionT, int GroupSizeT>
void templated_ParseTimestampYYYYmmDD()
{
    using Tokens = mp_list<
        YearDigit_c<4>,
        Text<mp_string<'-'>>,
        MonthDigit_c<2>,
        Text<mp_string<'-'>>,
        DayDigit_c<2>
    >;
    DatetimeTestContext<TimestampResolutionT> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetDateFormat("%Y-%m-%d");
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<
            JDatetimeOptions::TimestampResolution,
            TimestampResolutionT
        >
    >;
    using BA = JDatetime<Tokens, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

template<class TimestampResolutionT, int GroupSizeT>
void templated_ParseTimestampYYYYmmDDHHMMSS()
{
    using Tokens = mp_list<
            YearDigit_c<4>,
            Text<mp_string<'-'>>,
            MonthDigit_c<2>,
            Text<mp_string<'-'>>,
            DayDigit_c<2>,
            Text<mp_string<' '>>,
            HourDigit_c<2>,
            Text<mp_string<':'>>,
            MinuteDigit_c<2>,
            Text<mp_string<':'>>,
            SecondDigit_c<2>
    >;
    DatetimeTestContext<TimestampResolutionT> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetDateFormat("%Y-%m-%d %H:%M:%S");
    context.Initialize();
    using Options = boost::mp11::mp_list<
            boost::mp11::mp_list<
                    JDatetimeOptions::TimestampResolution,
                    TimestampResolutionT
            >
    >;
    using BA = JDatetime<Tokens, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

using Seconds = JDatetimeOptions::TimestampResolution::Seconds;
using Milliseconds = JDatetimeOptions::TimestampResolution::Milliseconds;

#define META_ParseJTimestampTestFixedYYYYmmDD(WS, TYPE)\
TEST_F(ParseJDataTest, Timestamp_YYYYmmDD_##TYPE##_W##WS) {\
	templated_ParseTimestampYYYYmmDD<TYPE, WS>();\
}

#define META_ParseJTimestampTestFixedYYYYmmDDHHMMSS(WS, TYPE)\
TEST_F(ParseJDataTest, Timestamp_YYYYmmDDHHMMSS_##TYPE##_W##WS) {\
	templated_ParseTimestampYYYYmmDDHHMMSS<TYPE, WS>();\
}

META_WS_4(META_ParseJTimestampTestFixedYYYYmmDD, Seconds)
META_WS_4(META_ParseJTimestampTestFixedYYYYmmDD, Milliseconds)
META_WS_4(META_ParseJTimestampTestFixedYYYYmmDDHHMMSS, Seconds)
META_WS_4(META_ParseJTimestampTestFixedYYYYmmDDHHMMSS, Milliseconds)

