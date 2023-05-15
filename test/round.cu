#include <gtest/gtest.h>
#include <meta_json_parser/action/jrealnumber.cuh>
#include <meta_json_parser/action/number_functors/round.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/number_test_context.cuh"
#include "test_utility/test_launcher.cuh"

class RoundTest : public ::testing::Test { };

template <intmax_t Dec, class NumT>
class RoundTestContext : public NumberTestContext<NumT> {
    using Base = NumberTestContext<NumT>;
    using Calculation = typename Base::Calculation;
public:
    RoundTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : NumberTestContext<NumT>(testSize, groupSize, seed) { }

    void InsertedNumberCallback(size_t index, Calculation value) override {
        Calculation val = static_cast<Calculation>(value);
        Calculation div = static_cast<Calculation>(pow(10.0, Dec));
        val = trunc(val*div)/div;
        (Base::m_h_correct)[index] = val;
    }
};

template<class OutTypeT, intmax_t DecT, int GroupSizeT>
void templated_Round()
{
    constexpr OutTypeT MinGenerate = -1000.0;
    constexpr OutTypeT MaxGenerate = 1000.0;
    using Functor = RoundFunctor<DecT, OutTypeT>;
    RoundTestContext<DecT, typename Functor::Num> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetMinimumValue(MinGenerate);
    context.SetMaximumValue(MaxGenerate);
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<JRealNumberOptions::JRealNumberTransformer, Functor>
    >;
    using BA = JRealNumber<OutTypeT, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_ParseJRealNumberTests(WS)\
TEST_F(RoundTest, float_W##WS) {\
	templated_Round<float, 1, WS>();\
}\
TEST_F(RoundTest, double_W##WS) {\
	templated_Round<double, 1, WS>();\
}\

META_WS_4(META_ParseJRealNumberTests)
