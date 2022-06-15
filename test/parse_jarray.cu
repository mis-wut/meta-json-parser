#include <gtest/gtest.h>
#include <random>
#include <boost/mp11.hpp>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jarray.cuh>
#include "test_utility/contexts/format_uint_2_test_context.cuh"
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"

using namespace boost::mp11;

template<class T>
class ParseJArray2UIntSuite : public ::testing::Test {};
TYPED_TEST_SUITE_P(ParseJArray2UIntSuite);

template<class T>
class ParseJArrayWorkgroupSuite : public ::testing::Test {};
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

    FormatUint2TestContext<OutType1, OutType2> context("[ %llu, %llu ]", TEST_SIZE, WorkGroupSize::value, SEED);
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

    FormatUint2TestContext<OutType1, OutType2> context("[ [ %llu],[[%llu] ]]", TEST_SIZE, WorkGroupSize::value, SEED);
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

    FormatUint2TestContext<uint32_t, uint64_t> context(
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

