#ifndef META_JSON_PARSER_ROUND_CUH
#define META_JSON_PARSER_ROUND_CUH
#include <type_traits>
#include <cmath>
#include <cuda_runtime_api.h>
 
 /**
 * @tparam Dec represents number of decimal places kept (non-negative integer)
 * @tparam NumT must be a floating point type
 */
template <intmax_t Dec, class NumT = float>
struct RoundFunctor {
    using Num = NumT;
    static_assert(std::is_floating_point<Num>::value, "NumT must be a floating point type.");
    static_assert(Dec >= 0, "Cannot round to negative decimal places");
 
    template <class NumberT> inline __device__ Num operator()(NumberT c) const {
        Num val = static_cast<Num>(c);
        Num div = static_cast<Num>(pow(10.0, Dec));
        val = trunc(val*div)/div;
        return val;
    }
};

#endif //META_JSON_PARSER_ROUND_CUH