#ifndef META_JSON_PARSER_TO_BOOL_CUH
#define META_JSON_PARSER_TO_BOOL_CUH
#include <type_traits>
#include <cuda_runtime_api.h>
 
 /**
 * @tparam InT must be an arithmetic type
 * @tparam OuT must be an integral type (default bool)
 */
template <class InT>
struct ToBoolFunctor {
    using In = InT;
 
    static_assert(std::is_arithmetic<In>::value, "InT must be an arithmetic type.");
 
    constexpr static In Zero = static_cast<In>(0);
 
    template <class NumberT> inline __device__ bool operator()(NumberT c) const {
        In val = static_cast<In>(c);
        return val > Zero;
    }
};

#endif //META_JSON_PARSER_TO_BOOL_CUH