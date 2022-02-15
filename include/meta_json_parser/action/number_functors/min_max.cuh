#ifndef META_JSON_PARSER_MIN_MAX_CUH
#define META_JSON_PARSER_MIN_MAX_CUH
#include <type_traits>
#include <ratio>
#include <cuda_runtime_api.h>

/**
 * @tparam MinT must be a type of std::ratio
 * @tparam MaxT must be a type of std::ratio
 */
template<class MinT, class MaxT, class OutT = float>
struct MinMaxNumberFunctor {
    using Min = MinT;
    using Max = MaxT;

    using OutType = OutT;
    static_assert(std::is_floating_point<OutType>::value, "OutT must be floating point type.");

    constexpr static OutType MinVal = static_cast<OutType>(Min::num) / static_cast<OutType>(Min::den);
    constexpr static OutType MaxVal = static_cast<OutType>(Max::num) / static_cast<OutType>(Max::den);
    constexpr static OutType Zero = static_cast<OutType>(0);
    constexpr static OutType One = static_cast<OutType>(1);
    static_assert(MinVal < MaxVal, "Min value must be lesser than Max value.");

    template<class NumberT>
    inline __device__ OutType operator()(NumberT c) const {
        OutType val = (static_cast<OutType>(c) - MinVal) / (MaxVal - MinVal);
        if (val < Zero)
            return Zero;
        else if (val > One)
            return One;
        return val;
    }
};

template<intmax_t MinNumerator, intmax_t MinDenominator,
        intmax_t MaxNumerator, intmax_t MaxDenominator, class OutT = float>
using MinMaxNumberFunctor_c = MinMaxNumberFunctor<
    std::ratio<MinNumerator, MinDenominator>,
    std::ratio<MaxNumerator, MaxDenominator>,
    OutT
>;


#endif //META_JSON_PARSER_MIN_MAX_CUH
