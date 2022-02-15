#include <gtest/gtest.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/number_functors/min_max.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/number_test_context.cuh"
#include "test_utility/test_launcher.cuh"

class MinMaxTest : public ::testing::Test { };

template<class MinT, class MaxT, class T>
class MinMaxTestContext : public NumberTestContext<T, uint64_t> {
    using Base = NumberTestContext<T, uint64_t>;
    using Calculation = typename Base::Calculation;
    using Min = MinT;
    using Max = MaxT;
    constexpr static Calculation MinVal = static_cast<Calculation>(Min::num) / static_cast<Calculation>(Min::den);
    constexpr static Calculation MaxVal = static_cast<Calculation>(Max::num) / static_cast<Calculation>(Max::den);
    constexpr static Calculation Zero = static_cast<Calculation>(0);
    constexpr static Calculation One = static_cast<Calculation>(1);
public:
    MinMaxTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : NumberTestContext<T, uint64_t>(testSize, groupSize, seed) { }

    void InsertedNumberCallback(size_t index, Calculation value) override {
        Calculation val = (static_cast<Calculation>(value) - MinVal) / (MaxVal - MinVal);
        if (val < Zero)
            val = Zero;
        else if (val > One)
            val = One;
        (*Base::m_h_correct)[index] = val;
    }
};

template<class OutTypeT, int GroupSizeT>
void templated_MinMax()
{
    constexpr size_t MinScale = 50'000;
    constexpr size_t MaxScale = 150'000;
    constexpr size_t MinGenerate = 0;
    constexpr size_t MaxGenerate = 200'000;
    using Functor = MinMaxNumberFunctor_c<MinScale, 1, MaxScale, 1, OutTypeT>;
    MinMaxTestContext<typename Functor::Min, typename Functor::Max, OutTypeT> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetMinimumValue(MinGenerate);
    context.SetMaximumValue(MaxGenerate);
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<JNumberOptions::JNumberTransformer, Functor>
    >;
    using BA = JNumber<uint64_t, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_ParseJNumberTests(WS)\
TEST_F(MinMaxTest, float_W##WS) {\
	templated_MinMax<float, WS>();\
}\
TEST_F(MinMaxTest, double_W##WS) {\
	templated_MinMax<double, WS>();\
}

META_WS_4(META_ParseJNumberTests)
