#include <string>
#include <gtest/gtest.h>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"
#include "test_utility/contexts/datetime_test_context.cuh"
#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/meta_utility/metastring.h>

class ParseJDateTest : public ::testing::Test { };

using namespace JsonParsers::DatetimeTokens;
using namespace boost::mp11;

template<class TimestampResolutionT, class Format, int GroupSizeT>
void templated_ParseTimestamp()
{
    using MetaString = typestring_to_metastring<Format>;
    DatetimeTestContext<TimestampResolutionT> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetDateFormat(Format::data());
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<
            JDatetimeOptions::TimestampResolution,
            TimestampResolutionT
        >
    >;
    using BA = JDatetime<MetaString, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

using Seconds = JDatetimeOptions::TimestampResolution::Seconds;
using Milliseconds = JDatetimeOptions::TimestampResolution::Milliseconds;

#define META_ParseJTimestampTest(WS, TYPE, FMT, NAME)\
TEST_F(ParseJDateTest, Timestamp_##NAME##_##TYPE##_W##WS) {\
	templated_ParseTimestamp<TYPE, typestring_is(FMT), WS>();\
}

META_WS_4(META_ParseJTimestampTest, Seconds, "%Y-%m-%d", YYYYmmdd)
META_WS_4(META_ParseJTimestampTest, Seconds, "%Y-%m-%d %H:%M:%S", YYYYmmddHHMMSS)
META_WS_4(META_ParseJTimestampTest, Milliseconds, "%Y-%m-%d", YYYYmmdd)
META_WS_4(META_ParseJTimestampTest, Milliseconds, "%Y-%m-%d %H:%M:%S", YYYYmmddHHMMSS)

