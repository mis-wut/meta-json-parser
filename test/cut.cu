#include <vector>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <boost/mp11/function.hpp>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/number_functors/cut.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/number_test_context.cuh"
#include "test_utility/test_launcher.cuh"

using namespace boost::mp11;

template<typename T>
class CutTest : public ::testing::Test { };

template<class CutFunctorT, class GenerateT>
class CutTestContext : public NumberTestContext<typename CutFunctorT::OutType, GenerateT, GenerateT> {
    using Base = NumberTestContext<typename CutFunctorT::OutType, GenerateT, GenerateT>;
    using Functor = CutFunctorT;
    using Calculation = typename Base::Calculation;
    using Bins = typename Functor::Bins;
    using Labels = typename Functor::Labels;
    using LabelType = typename Functor::OutType;
protected:
    std::vector<Calculation> m_bins;
    std::vector<LabelType> m_labels;
public:
    CutTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : NumberTestContext<typename Functor::OutType, GenerateT, GenerateT>(testSize, groupSize, seed),
            m_bins(mp_size<Bins>::value),
            m_labels(mp_size<Labels>::value) {
        auto bin_it = m_bins.begin();
        mp_for_each<Bins>([&](auto bin) {
            using Bin = decltype(bin);
            *bin_it++ = Bin::GetValue();
        });
        auto label_it = m_labels.begin();
        mp_for_each<Labels>([&](auto label) {
            using Label = decltype(label);
            *label_it++ = Label::GetValue();
        });
    }

    void InsertedNumberCallback(size_t index, Calculation value) override {
        using Right = typename Functor::RightOpt;
        std::remove_reference_t<decltype(m_bins.front())> val;
        if ((Right::value && (value <= m_bins.front() || value > m_bins.back())) ||
            (!Right::value && (value < m_bins.front() || value >= m_bins.back()))) {
            if constexpr (mp_similar<typename Functor::OutOfRangeOpt, CutFunctorOptions::OutOfRange::ConstValue<mp_int<0>>>::value) {
                using Value = mp_first<typename Functor::OutOfRangeOpt>;
                val = static_cast<decltype(val)>(Value::value);
            } else if constexpr(mp_same<typename Functor::OutOfRangeOpt, CutFunctorOptions::OutOfRange::NanValue>::value) {
                val = NAN;
            }
        } else {
            decltype(m_bins.begin()) bin_it;
            if constexpr (Right::value) {
                bin_it = std::lower_bound(m_bins.begin(), m_bins.end(), value);
            } else {
                bin_it = std::upper_bound(m_bins.begin(), m_bins.end(), value);
            }
            val = m_labels[bin_it - m_bins.begin() - 1];
        }
        (Base::m_h_correct)[index] = val;
    }
};

// -100, -90, ... 90, 100
using Bins_100To100 = mp_transform_q<
    mp_compose_q<
        mp_bind_front<
            mp_plus,
            mp_int<-10>
        >,
        mp_bind_front<
            mp_mul,
            mp_int<10>
        >,
        mp_quote<CutBin>
    >,
    mp_iota<mp_int<21>>
>;

TYPED_TEST_SUITE_P(CutTest);

TYPED_TEST_P(CutTest, InRange) {
    using WorkGroupSize = TypeParam;
    using Bins = Bins_100To100;
    using FunctorOpts = mp_list<
        mp_list<
            CutFunctorOptions::OutOfRange,
            CutFunctorOptions::OutOfRange::ConstValue<mp_int<-99999>>
        >
    >;
    using Functor = CutFunctor<Bins, FunctorOpts>;

    CutTestContext<Functor, int32_t> context(TEST_SIZE, WorkGroupSize::value, SEED);
    context.SetMinimumValue(-99);
    context.SetMaximumValue(100);
    context.Initialize();
    using Options = mp_list<
        mp_list<JNumberOptions::JNumberTransformer, Functor>,
        mp_list<JNumberOptions::JNumberSign, JNumberOptions::JNumberSign::Signed> >;
    using BA = JNumber<int32_t, void, Options>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(CutTest, OpenRight) {
    using WorkGroupSize = TypeParam;
    using Bins = Bins_100To100;
    using FunctorOpts = mp_list<
        mp_list<
            CutFunctorOptions::Right,
            CutFunctorOptions::Right::False
        >,
        mp_list<
            CutFunctorOptions::OutOfRange,
            CutFunctorOptions::OutOfRange::ConstValue<mp_int<-99999>>
        >
    >;
    using Functor = CutFunctor<Bins, FunctorOpts>;

    CutTestContext<Functor, int32_t> context(TEST_SIZE, WorkGroupSize::value, SEED);
    context.SetMinimumValue(-100);
    context.SetMaximumValue(99);
    context.Initialize();
    using Options = mp_list<
            mp_list<JNumberOptions::JNumberTransformer, Functor>,
            mp_list<JNumberOptions::JNumberSign, JNumberOptions::JNumberSign::Signed>
    >;
    using BA = JNumber<int32_t, void, Options>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(CutTest, CustomLabels) {
    using WorkGroupSize = TypeParam;
    using Bins = mp_list<CutBinInt32_c<-10>, CutBinInt32_c<-5>, CutBinInt32_c<0>, CutBinInt32_c<5>, CutBinInt32_c<10>>;
    using Labels = mp_list<
        CutLabel<std::integral_constant<char, 'A'>>,
        CutLabel<std::integral_constant<char, 'B'>>,
        CutLabel<std::integral_constant<char, 'C'>>,
        CutLabel<std::integral_constant<char, 'D'>>
    >;
    using FunctorOpts = mp_list<
        mp_list<
            CutFunctorOptions::OutOfRange,
            CutFunctorOptions::OutOfRange::ConstValue<std::integral_constant<char, 'X'>>
        >,
        mp_list<
            CutFunctorOptions::Labels,
            Labels
        >
    >;
    using Functor = CutFunctor<Bins, FunctorOpts>;

    CutTestContext<Functor, int32_t> context(TEST_SIZE, WorkGroupSize::value, SEED);
    context.SetMinimumValue(-12);
    context.SetMaximumValue(12);
    context.Initialize();
    using Options = mp_list<
        mp_list<JNumberOptions::JNumberTransformer, Functor>,
        mp_list<JNumberOptions::JNumberSign, JNumberOptions::JNumberSign::Signed>
    >;
    using BA = JNumber<int32_t, void, Options>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(CutTest, OutOfRangeConstValue) {
    using WorkGroupSize = TypeParam;
    using Bins = mp_list<CutBinInt32_c<-10>, CutBinInt32_c<-5>, CutBinInt32_c<0>, CutBinInt32_c<5>, CutBinInt32_c<10>>;
    using FunctorOpts = mp_list<
        mp_list<
            CutFunctorOptions::OutOfRange,
            CutFunctorOptions::OutOfRange::ConstValue<mp_int<-99999>>
        >
    >;
    using Functor = CutFunctor<Bins, FunctorOpts>;

    CutTestContext<Functor, int32_t> context(TEST_SIZE, WorkGroupSize::value, SEED);
    context.SetMinimumValue(-3000);
    context.SetMaximumValue(3000);
    context.Initialize();
    using Options = mp_list<
        mp_list<JNumberOptions::JNumberTransformer, Functor>,
        mp_list<JNumberOptions::JNumberSign, JNumberOptions::JNumberSign::Signed>
    >;
    using BA = JNumber<int32_t, void, Options>;
    LaunchTest<BA, WorkGroupSize>(context);
}

TYPED_TEST_P(CutTest, OutOfRangeNan) {
    using WorkGroupSize = TypeParam;
    using Bins = mp_list<CutBinReal_c<-10, 1>, CutBinReal_c<-5, 1>, CutBinReal_c<0, 1>, CutBinReal_c<5, 1>, CutBinReal_c<10, 1>>;
    using FunctorOpts = mp_list<
        mp_list<
            CutFunctorOptions::OutOfRange,
            CutFunctorOptions::OutOfRange::NanValue
        >
    >;
    using Functor = CutFunctor<Bins, FunctorOpts>;

    CutTestContext<Functor, int32_t> context(TEST_SIZE, WorkGroupSize::value, SEED);
    context.SetMinimumValue(-3000);
    context.SetMaximumValue(3000);
    context.Initialize();
    using Options = mp_list<
        mp_list<JNumberOptions::JNumberTransformer, Functor>,
        mp_list<JNumberOptions::JNumberSign, JNumberOptions::JNumberSign::Signed>
    >;
    using BA = JNumber<int32_t, void, Options>;
    LaunchTest<BA, WorkGroupSize>(context);
}

REGISTER_TYPED_TEST_SUITE_P(CutTest, InRange, OpenRight, CustomLabels, OutOfRangeConstValue, OutOfRangeNan);

using Types = mp_rename<AllWorkGroups, ::testing::Types>;

INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroups, CutTest, Types, WorkGroupNameGenerator);
