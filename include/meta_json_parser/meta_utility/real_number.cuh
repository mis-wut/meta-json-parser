#ifndef META_JSON_PARSER_REAL_NUMBER_H
#define META_JSON_PARSER_REAL_NUMBER_H
#include <ratio>
#include <type_traits>

template<class NumT, class DenT, class OutTypeT = float>
struct RealNumber : public std::ratio<NumT::value, DenT::value> {
    using OutType = OutTypeT;

    __forceinline__ __device__ constexpr static OutType GetValue() {
        return static_cast<OutType>(NumT::value) / static_cast<OutType>(DenT::value);
    };
};

template<intmax_t Num, intmax_t Den, class OutTypeT = float>
using RealNumber_c = RealNumber<
    std::integral_constant<intmax_t, Num>,
    std::integral_constant<intmax_t, Den>,
    OutTypeT
>;

#endif //META_JSON_PARSER_REAL_NUMBER_H
