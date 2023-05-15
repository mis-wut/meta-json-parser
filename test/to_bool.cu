#include <gtest/gtest.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/number_functors/to_bool.cuh>
#include "test_helper.h"
#include "test_configuration.h"
#include "test_utility/contexts/number_test_context.cuh"
#include "test_utility/test_launcher.cuh"

class ToBoolTest : public ::testing::Test { };

template <class InT>
class ToBoolTestContext : public NumberTestContext<InT> {
    using Base = NumberTestContext<InT>;
    using Calculation = typename Base::Calculation;
    constexpr static InT Zero = static_cast<InT>(0);
public:
    ToBoolTestContext(size_t testSize, size_t groupSize, unsigned long seed)
            : NumberTestContext<InT>(testSize, groupSize, seed) { }

    void InsertedNumberCallback(size_t index, Calculation value) override {
        Calculation result = static_cast<Calculation>(value > Zero);
        (Base::m_h_correct)[index] = result;
    }
};

template<class OutTypeT, int GroupSizeT>
void templated_ToBool()
{
    constexpr OutTypeT MinGenerate = -1000;
    constexpr OutTypeT MaxGenerate = 1000;
    using Functor = ToBoolFunctor<OutTypeT>;
    ToBoolTestContext<typename Functor::In> context(TEST_SIZE, GroupSizeT, SEED);
    context.SetMinimumValue(MinGenerate);
    context.SetMaximumValue(MaxGenerate);
    context.Initialize();
    using Options = boost::mp11::mp_list<
        boost::mp11::mp_list<JNumberOptions::JNumberTransformer, Functor>
    >;
    using BA = JNumber<OutTypeT, void, Options>;
    LaunchTest<BA, boost::mp11::mp_int<GroupSizeT>>(context);
}

#define META_ParseJBooleanTests(WS)\
TEST_F(ToBoolTest, int32_W##WS) {\
	templated_ToBool<int32_t, WS>();\
}\
META_WS_4(META_ParseJBooleanTests)
