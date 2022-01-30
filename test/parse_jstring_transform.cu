#include <algorithm>
#include <string>
#include <cctype>
#include <boost/mp11/integral.hpp>
#include <gtest/gtest.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/action/jstring_custom.cuh>
#include <meta_json_parser/action/string_functors/letter_case_functors.cuh>
#include "test_helper.h"
#include "test_utility/contexts/string_test_context.cuh"
#include "test_configuration.h"
#include "test_utility/test_launcher.cuh"
#include "test_utility/contexts/static_string_test_context.cuh"

class ParseJStringTransformTest : public ::testing::Test { };

template<class FunctorT>
class FunctorTestContext : public StaticStringTestContext {
    FunctorT m_functor;
protected:
    void InsertedWordCallback(size_t index, std::string_view word) override {
        std::vector<char> upper_word(std::max(word.length(), (size_t)m_static_string_size), '\0');
        std::transform(word.begin(), word.end(), upper_word.begin(), m_functor);
        snprintf(
            m_h_correct.data() + index * m_static_string_size,
            m_static_string_size + 1,
            "%s",
            upper_word.data()
        );
    }

public:
    FunctorTestContext(FunctorT functor, size_t test_size, size_t group_size, SeedType seed, uint32_t static_string_size)
            : StaticStringTestContext(test_size, group_size, seed, static_string_size), m_functor(functor) { }
};

template<class FunctorT, class SeedTypeT>
FunctorTestContext<FunctorT> MakeFunctorTestContext(
    FunctorT functor, size_t test_size, size_t group_size, SeedTypeT seed, uint32_t static_string_size
) {
    return FunctorTestContext<FunctorT>(functor, test_size, group_size, seed, static_string_size);
}

void SetupLowerUpperCase(StaticStringTestContext& context) {
    std::vector<char> allowed_chars;
    auto inserter = std::back_inserter(allowed_chars);
    std::generate_n(inserter, 'z' - 'a' + 1, [c='a']() mutable { return c++; });
    std::generate_n(inserter, 'Z' - 'A' + 1, [c='A']() mutable { return c++; });
    std::generate_n(inserter, 10, [c='0']() mutable { return c++; });
    *inserter++ = '#';
    *inserter++ = '$';
    *inserter++ = '%';
    context.SetMinimumLength(1);
    context.SetMaximumLength(65);
    context.SetAllowedChars(std::move(allowed_chars));
}

template<int CopyBytesT, int GroupSizeT>
void templated_ToUpper() {
    auto context = MakeFunctorTestContext(
        [](char c) { return (char)std::toupper(c); },
        TEST_SIZE,
        GroupSizeT,
        SEED,
        CopyBytesT
    );
    SetupLowerUpperCase(context);
    context.SetMaximumEscapedCharacters(8);
    context.Initialize();
    using BaseAction = JStringStaticCopy<
            boost::mp11::mp_int<CopyBytesT>,
            int,
            boost::mp11::mp_list<
                boost::mp11::mp_list<JStringOptions::JStringCharTransformer, ToUpperStringTransformer>
            >
    >;
    LaunchTest<BaseAction, boost::mp11::mp_int<GroupSizeT>>(context);
}

template<int CopyBytesT, int GroupSizeT>
void templated_ToLower() {
    auto context = MakeFunctorTestContext(
            [](char c) { return (char)std::tolower(c); },
            TEST_SIZE,
            GroupSizeT,
            SEED,
            CopyBytesT
    );
    SetupLowerUpperCase(context);
    context.SetMaximumEscapedCharacters(8);
    context.Initialize();
    using BaseAction = JStringStaticCopy<
            boost::mp11::mp_int<CopyBytesT>,
            int,
            boost::mp11::mp_list<
                    boost::mp11::mp_list<JStringOptions::JStringCharTransformer, ToLowerStringTransformer>
            >
    >;
    LaunchTest<BaseAction, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_jstring_custom_tests(WS)\
TEST_F(ParseJStringTransformTest, to_upper_static_copy_B5_W##WS) {\
    templated_ToUpper<5, WS>();\
}\
TEST_F(ParseJStringTransformTest, to_upper_static_copy_B65##WS) {\
    templated_ToUpper<65, WS>();\
}\
TEST_F(ParseJStringTransformTest, to_lower_static_copy_B5_W##WS) {\
    templated_ToLower<5, WS>();\
}\
TEST_F(ParseJStringTransformTest, to_lower_static_copy_B65##WS) {\
    templated_ToLower<65, WS>();\
}

META_WS_4(META_jstring_custom_tests)
