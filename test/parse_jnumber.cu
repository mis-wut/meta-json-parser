#include <gtest/gtest.h>
#include <meta_json_parser/action/jnumber.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/number_test_context.cuh"
#include "test_utility/test_launcher.cuh"

class ParseJNumberTest : public ::testing::Test { };

template<class OutTypeT, int GroupSizeT>
void templated_ParseInteger()
{
    NumberTestContext<OutTypeT> context(TEST_SIZE, GroupSizeT, SEED);
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<
            JNumberOptions::JNumberSign,
            boost::mp11::mp_if<
                std::is_signed<OutTypeT>,
                JNumberOptions::JNumberSign::Signed,
                JNumberOptions::JNumberSign::Unsigned
            >
        >
    >;
    using BA = JNumber<OutTypeT, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_ParseJNumberTests(WS)\
TEST_F(ParseJNumberTest, uint32_W##WS) {\
	templated_ParseInteger<uint32_t, WS>();\
}\
TEST_F(ParseJNumberTest, uint64_W##WS) {\
	templated_ParseInteger<uint64_t, WS>();\
}\
TEST_F(ParseJNumberTest, uint16_W##WS) {\
	templated_ParseInteger<uint16_t, WS>();\
}\
TEST_F(ParseJNumberTest, uint8_W##WS) {\
	templated_ParseInteger<uint8_t, WS>();\
}\
TEST_F(ParseJNumberTest, int32_W##WS) {\
	templated_ParseInteger<int32_t, WS>();\
}\
TEST_F(ParseJNumberTest, int64_W##WS) {\
	templated_ParseInteger<int64_t, WS>();\
}\
TEST_F(ParseJNumberTest, int16_W##WS) {\
	templated_ParseInteger<int16_t, WS>();\
}\
TEST_F(ParseJNumberTest, int8_W##WS) {\
	templated_ParseInteger<int8_t, WS>();\
}

META_WS_4(META_ParseJNumberTests)
