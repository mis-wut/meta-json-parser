#include <gtest/gtest.h>
#include <random>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/meta_utility/map_utility.h>
#include <meta_json_parser/meta_utility/metastring.h>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/format_uint_2_test_context.cuh"
#include "test_utility/test_launcher.cuh"

using namespace boost::mp11;

template<class T>
class ParseJDict2UIntSuite : public ::testing::Test { };
TYPED_TEST_SUITE_P(ParseJDict2UIntSuite);

template<class T>
class ParseJDictWorkgroupSuite : public ::testing::Test {};
TYPED_TEST_SUITE_P(ParseJDictWorkgroupSuite);

TYPED_TEST_P(ParseJDict2UIntSuite, ParseDict2UInt) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using OutTypes = mp_at_c<TypeParam, 1>;
    using OutType1 = mp_at_c<OutTypes, 0>;
    using OutType2 = mp_at_c<OutTypes, 1>;
    using Opts = mp_at_c<TypeParam, 2>;
    using DictOptions = MapEntries<
        JDictOpts::Order, mp_first<Opts>,
        JDictOpts::Skip, mp_second<Opts>
    >;
    using Key1 = metastring("Key num one");
    using Key2 = metastring("Second key");
    using BA = JDict<
        DictEntries<
            Key1, JNumber<OutType1, Key1>,
            Key2, JNumber<OutType2, Key2>
        >,
        DictOptions
    >;

    std::vector<Uint2Format> formats;
    formats.emplace_back( R"JSON({ "Key num one": %llu, "Second key": %llu })JSON");
    if (std::is_same_v<mp_first<Opts>, JDictOpts::Order::RandomOrder>)
        formats.emplace_back(
            R"JSON({ "Second key": %llu, "Key num one": %llu })JSON",
            Uint2Format::Order::SecondFirst
        );

    FormatUint2TestContext<OutType1, OutType2> context(formats, TEST_SIZE, WorkGroupSize::value, SEED);
    context.Initialize();
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(ParseJDict2UIntSuite, ParseNestedDicts) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using OutTypes = mp_at_c<TypeParam, 1>;
    using OutType1 = mp_at_c<OutTypes, 0>;
    using OutType2 = mp_at_c<OutTypes, 1>;
    using Opts = mp_at_c<TypeParam, 2>;
    using DictOptions = MapEntries<
        JDictOpts::Order, mp_first<Opts>,
        JDictOpts::Skip, mp_second<Opts>
    >;
    using Key1 = metastring("Key num one");
    using Key2 = metastring("Second key");
    using Nested = metastring("Nested");
    using Nested1 = metastring("Nested1");
    using Nested2 = metastring("Nested2");
    using BA = JDict<
        DictEntries<
            Nested, JDict<
                DictEntries<
                    Nested1, JDict<
                        DictEntries<
                            Key1, JNumber<OutType1, Key1>
                        >,
                        DictOptions
                    >,
                    Nested2, JDict<
                        DictEntries<
                            Nested, JDict<
                                DictEntries<
                                    Key2, JNumber<OutType2, Key2>
                                >,
                                DictOptions
                            >
                        >,
                        DictOptions
                    >
                >,
                DictOptions
            >
        >,
        DictOptions
    >;

    std::vector<Uint2Format> formats;
    formats.emplace_back(R"JSON({ "Nested": { "Nested1": { "Key num one": %llu } , "Nested2": { "Nested": {"Second key": %llu} }}})JSON");
    if (std::is_same_v<mp_first<Opts>, JDictOpts::Order::RandomOrder>)
        formats.emplace_back(
            R"JSON({ "Nested": { "Nested2": { "Nested": {"Second key": %llu} }, "Nested1": { "Key num one": %llu }}})JSON",
            Uint2Format::Order::SecondFirst
        );

    FormatUint2TestContext<OutType1, OutType2> context(formats, TEST_SIZE, WorkGroupSize::value, SEED);
    context.Initialize();
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(ParseJDictWorkgroupSuite, Skipping) {
    using WorkGroupSize = TypeParam;
    using DictOptions = MapEntries<
        JDictOpts::Order, JDictOpts::Order::RandomOrder,
        JDictOpts::Skip, JDictOpts::Skip::Enable_c<8>
    >;
    using Key1 = metastring("Key num one");
    using Key2 = metastring("Second key");
    using Nested = metastring("Nested");
    using Nested1 = metastring("Nested1");
    using Nested2 = metastring("Nested2");
    using BA = JDict<
        DictEntries<
            Nested, JDict<
                DictEntries<
                    Nested1, JDict<
                        DictEntries<
                            Key1, JNumber<uint32_t, Key1>
                        >,
                        DictOptions
                    >,
                    Nested2, JDict<
                        DictEntries<
                            Nested, JDict<
                                DictEntries<
                                    Key2, JNumber<uint32_t, Key2>
                                >,
                                DictOptions
                            >
                        >,
                        DictOptions
                    >
                >,
                DictOptions
            >
        >,
        DictOptions
    >;

    FormatUint2TestContext<uint32_t, uint32_t> context({
           { R"JSON({ "Nested": { "Skip it": [ "nothing" ],"Nested1": { "Key num one": %llu, "Skip me": null } , "Nested2": { "Nested": {"Second key": %llu} }}})JSON" },
           { R"JSON({ "Do not bother": [ true, 12.32, false], "Nested": { "Nested2": { "Nested": {"null": 0.42, "Second key": %llu, "false": true} }, "Nested1": { "Key num one": %llu }}})JSON", Uint2Format::Order::SecondFirst }
       }, TEST_SIZE, WorkGroupSize::value, SEED);
    context.Initialize();
    LaunchTest<BA, WorkGroupSize>(context);
}

REGISTER_TYPED_TEST_SUITE_P(ParseJDict2UIntSuite, ParseDict2UInt, ParseNestedDicts);
REGISTER_TYPED_TEST_SUITE_P(ParseJDictWorkgroupSuite, Skipping);

using UnsignedTypes = mp_list<
    mp_list<uint8_t, uint64_t>,
    mp_list<uint32_t, uint16_t>,
    mp_list<uint16_t, uint8_t>
>;

using DictOption = mp_list<
    mp_list<JDictOpts::Order::RandomOrder, JDictOpts::Skip::Disable>,
    mp_list<JDictOpts::Order::RandomOrder, JDictOpts::Skip::Enable_c<8>>,
    mp_list<JDictOpts::Order::ConstOrder, JDictOpts::Skip::Disable>
>;

using AllWorkGroupsWith2UIntTypes = mp_rename<mp_product<
    mp_list,
    AllWorkGroups,
    UnsignedTypes,
    DictOption
>, ::testing::Types>;

using AllWorkGroupsTypes = mp_rename<AllWorkGroups, ::testing::Types>;

struct NameGenerator {
    template <typename TypeParam>
    static std::string GetName(int i) {
        using GroupSize = mp_at_c<TypeParam, 0>;
        using OutTypes = mp_at_c<TypeParam, 1>;
        using OutType1 = mp_at_c<OutTypes, 0>;
        using OutType2 = mp_at_c<OutTypes, 1>;
        using Opts = mp_at_c<TypeParam, 2>;
        using Order = mp_first<Opts>;
        using Skipping = mp_second<Opts>;

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

        if constexpr (std::is_same_v<Skipping , JDictOpts::Skip::Disable>) {
            stream << "disable_skip";
        } else {
            stream << "enable_skip";
        }

        stream << "_";

        if constexpr (std::is_same_v<Order, JDictOpts::Order::ConstOrder>) {
            stream << "const_order";
        } else {
            stream << "random_order";
        }

        return stream.str();
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroupsWith2UInt, ParseJDict2UIntSuite, AllWorkGroupsWith2UIntTypes, NameGenerator);
INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroups, ParseJDictWorkgroupSuite, AllWorkGroupsTypes, WorkGroupNameGenerator);
