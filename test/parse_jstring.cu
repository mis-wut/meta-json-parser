#include <gtest/gtest.h>
#include <boost/mp11/integral.hpp>
#include <meta_json_parser/action/jstring.cuh>
#include "test_helper.h"
#include "test_utility/contexts/static_string_test_context.cuh"
#include "test_utility/test_launcher.cuh"
#include "test_configuration.h"

class ParseJStringTest : public ::testing::Test { };

template<int GroupSizeT>
void templated_ValidateString(StringTestContext& context) {
    context.Initialize();
    using BaseAction = JString;
    LaunchTest<BaseAction, boost::mp11::mp_int<GroupSizeT>>(context);
}

template<int GroupSizeT>
void templated_ValidateSimpleString() {
    StringTestContext context(TEST_SIZE, GroupSizeT, SEED);
    context.SetMinimumLength(1);
    context.SetMaximumLength(65);
    context.SetMaximumEscapedCharacters(0);
    templated_ValidateString<GroupSizeT>(context);
}

template<int GroupSizeT>
void templated_ValidateEscapedString() {
    StringTestContext context(TEST_SIZE, GroupSizeT, SEED);
    context.SetMinimumLength(1);
    context.SetMaximumLength(65);
    context.SetMaximumEscapedCharacters(8);
    templated_ValidateString<GroupSizeT>(context);
}

template<int CopyBytes, int GroupSizeT>
void templated_ParseStringStaticCopy() {
    StaticStringTestContext context(TEST_SIZE, GroupSizeT, SEED, CopyBytes);
    context.SetMinimumLength(CopyBytes / 2);
    context.SetMaximumLength(CopyBytes + 8);
    context.SetMaximumEscapedCharacters(4);
    context.Initialize();
    using BaseAction = JStringStaticCopy<boost::mp11::mp_int<CopyBytes>, char>;
    LaunchTest<BaseAction, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_jstring_tests(WS)\
TEST_F(ParseJStringTest, validation_W##WS) { \
    templated_ValidateSimpleString<WS>(); \
}\
TEST_F(ParseJStringTest, validation_backslash_W##WS) {\
    templated_ValidateEscapedString<WS>(); \
}\
TEST_F(ParseJStringTest, static_copy_B5_W##WS) {\
	templated_ParseStringStaticCopy<5, WS>();\
}\
TEST_F(ParseJStringTest, static_copy_B33_W##WS) {\
	templated_ParseStringStaticCopy<33, WS>();\
}\
TEST_F(ParseJStringTest, static_copy_B60_W##WS) {\
	templated_ParseStringStaticCopy<60, WS>();\
}

META_WS_4(META_jstring_tests)
