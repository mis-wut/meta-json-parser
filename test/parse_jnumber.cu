#include <gtest/gtest.h>
#include <meta_json_parser/action/jnumber.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/number_test_context.cuh"
#include "test_utility/test_launcher.cuh"

class ParseJNumberTest : public ::testing::Test { };

template<class OutTypeT, int GroupSizeT>
void templated_ParseUnsignedInteger()
{
    NumberTestContext<OutTypeT> context(TEST_SIZE, GroupSizeT, SEED);
    context.Initialize();
    using BA = JNumber<OutTypeT, void>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_ParseJNumberTests(WS)\
TEST_F(ParseJNumberTest, uint32_W##WS) {\
	templated_ParseUnsignedInteger<uint32_t, WS>();\
}\
TEST_F(ParseJNumberTest, uint64_W##WS) {\
	templated_ParseUnsignedInteger<uint64_t, WS>();\
}\
TEST_F(ParseJNumberTest, uint16_W##WS) {\
	templated_ParseUnsignedInteger<uint16_t, WS>();\
}\
TEST_F(ParseJNumberTest, uint8_W##WS) {\
	templated_ParseUnsignedInteger<uint8_t, WS>();\
}

META_WS_4(META_ParseJNumberTests)
