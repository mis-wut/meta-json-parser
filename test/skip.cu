#include <type_traits>
#include <gtest/gtest.h>
#include <boost/mp11/function.hpp>
#include <meta_json_parser/action/jarray.cuh>
#include <meta_json_parser/action/skip_action.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/repeat_test_context.cuh"
#include "test_utility/test_launcher.cuh"

using namespace boost::mp11;

template<typename T>
class SkipTestSingleTypeSuite : public ::testing::Test { };

template<typename T>
class SkipTestCombinedTypesSuite : public ::testing::Test { };

TYPED_TEST_SUITE_P(SkipTestSingleTypeSuite);
TYPED_TEST_SUITE_P(SkipTestCombinedTypesSuite);

TYPED_TEST_P(SkipTestSingleTypeSuite, String) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using LimitedSkipTypes = mp_at_c<TypeParam, 1>;
    using SkipTypes = mp_if<
        LimitedSkipTypes,
        mp_list<JsonParsers::SkipJsonTypes::String>,
        JsonParsers::SkipAllTypes
    >;

    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON("some string")JSON");
    context.AppendEntry(R"JSON("123436")JSON");
    context.AppendEntry(R"JSON("001.432")JSON");
    context.AppendEntry(R"JSON(".e432+")JSON");
    context.AppendEntry(R"JSON("null")JSON");
    context.AppendEntry(R"JSON("true")JSON");
    context.Initialize();

    using BA = SkipAction_c<4, SkipTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(SkipTestSingleTypeSuite, Number) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using LimitedSkipTypes = mp_at_c<TypeParam, 1>;
    using SkipTypes = mp_if<
        LimitedSkipTypes,
        mp_list<JsonParsers::SkipJsonTypes::Number>,
        JsonParsers::SkipAllTypes
    >;

    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON(34291045)JSON");
    context.AppendEntry(R"JSON(0)JSON");
    context.AppendEntry(R"JSON(0.00)JSON");
    context.AppendEntry(R"JSON(-0)JSON");
    context.AppendEntry(R"JSON(-2356.434)JSON");
    context.AppendEntry(R"JSON(2356.434e32)JSON");
    context.AppendEntry(R"JSON(2356.434e-32)JSON");
    context.AppendEntry(R"JSON(-2356.434E+32)JSON");
    context.Initialize();

    using BA = SkipAction_c<4, SkipTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(SkipTestSingleTypeSuite, Boolean) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using LimitedSkipTypes = mp_at_c<TypeParam, 1>;
    using SkipTypes = mp_if<
        LimitedSkipTypes,
        mp_list<JsonParsers::SkipJsonTypes::Boolean>,
        JsonParsers::SkipAllTypes
    >;

    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON(true)JSON");
    context.AppendEntry(R"JSON(false)JSON");
    context.Initialize();

    using BA = SkipAction_c<4, SkipTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(SkipTestSingleTypeSuite, Null) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using LimitedSkipTypes = mp_at_c<TypeParam, 1>;
    using SkipTypes = mp_if<
        LimitedSkipTypes,
        mp_list<JsonParsers::SkipJsonTypes::Null>,
        JsonParsers::SkipAllTypes
    >;

    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON(null)JSON");
    context.Initialize();

    using BA = SkipAction_c<4, SkipTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(SkipTestSingleTypeSuite, Array) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using LimitedSkipTypes = mp_at_c<TypeParam, 1>;
    using SkipTypes = mp_if<
        LimitedSkipTypes,
        mp_list<JsonParsers::SkipJsonTypes::Array>,
        JsonParsers::SkipAllTypes
    >;

    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON([])JSON");
    context.AppendEntry(R"JSON([ [[ []] ]])JSON");
    context.AppendEntry(R"JSON([[[],[]], [ ] , [[  ],[[]]]])JSON");
    context.AppendEntry(R"JSON([ [ [ [ ] ]] , [ ], [], [ ],    [  ],[[[   []]]] ] )JSON");
    context.Initialize();

    using BA = SkipAction_c<8, SkipTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(SkipTestSingleTypeSuite, Object) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using LimitedSkipTypes = mp_at_c<TypeParam, 1>;
    using SkipTypes = mp_if<
        LimitedSkipTypes,
        mp_list<JsonParsers::SkipJsonTypes::Array>,
        JsonParsers::SkipAllTypes
    >;

    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON({})JSON");
    context.AppendEntry(R"JSON({"am41_":{}})JSON");
    context.AppendEntry(R"JSON({"4123.V_q":{} , ",11111111111S"  :{}})JSON");
    context.AppendEntry(R"JSON({"askl;mm":{ "a;0cxma_+!": {"4sm 4ol"  : {}} } , "adk1_A+!"  :{}})JSON");
    context.Initialize();

    using BA = SkipAction_c<8, SkipTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(SkipTestCombinedTypesSuite, CombinedTypes) {
    using WorkGroupSize = TypeParam;
    RepeatTestContext context(TEST_SIZE, WorkGroupSize::value, SEED);

    context.AppendEntry(R"JSON({"x":[{"x":[{"x":[]}]}]})JSON");
    context.AppendEntry(R"JSON({"am41_":{"m2 3  3": [null, {"1245": 0.432e+23, "2g4": [true]}]}})JSON");
    context.AppendEntry(R"JSON(["t13", [ true, false, 0.6452, { "tg2e": [[null], 0.123e-12 ] } , "324", 324] , "235" ])JSON");
    context.Initialize();

    using BA = SkipAction_c<8, JsonParsers::SkipAllTypes>;
    LaunchTest<BA, WorkGroupSize>(context);
}

REGISTER_TYPED_TEST_SUITE_P(SkipTestSingleTypeSuite, String, Number, Boolean, Null, Array, Object);
REGISTER_TYPED_TEST_SUITE_P(SkipTestCombinedTypesSuite, CombinedTypes);

using SingleTypes = mp_rename<mp_product<
    mp_list,
    AllWorkGroups,
    mp_list<
        mp_true,
        mp_false
    >
>, ::testing::Types>;

using CombinedTypes = mp_rename<AllWorkGroups, ::testing::Types>;

struct NameGenerator {
    template <typename TypeParam>
    static std::string GetName(int i) {
        using GroupSize = mp_at_c<TypeParam, 0>;
        using LimitedSkipTypes = mp_at_c<TypeParam, 1>;

        std::stringstream stream;
        stream << "WS_" << GroupSize::value;
        stream << "_";

        if constexpr (LimitedSkipTypes::value) {
            stream << "limited_skip_types";
        } else {
            stream << "all_skip_types";
        }

        return stream.str();
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroups, SkipTestSingleTypeSuite, SingleTypes, NameGenerator);
INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroups, SkipTestCombinedTypesSuite, CombinedTypes, WorkGroupNameGenerator);
