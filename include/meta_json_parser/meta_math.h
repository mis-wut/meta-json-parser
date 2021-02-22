#pragma once
#include <boost/mp11/integral.hpp>
#include <boost/mp11/function.hpp>

template<typename ValueT>
struct IsPower2_impl
{
    using type = boost::mp11::mp_bool<!(ValueT::value&(ValueT::value - 1))>;
};

template<typename ValueT>
using IsPower2 = IsPower2_impl<ValueT>::type;

template<int ValueT>
using IsPower2_c = IsPower2_impl<boost::mp11::mp_int<ValueT>>::type;

namespace boost {
namespace mp11 {

template<class T1, class T2> using mp_equal = mp_bool<(T1::value == T2::value) && !((T1::value < 0 && T2::value >= 0) || (T1::value >= 0 && T2::value < 0))>;

template<class T1, class T2> using mp_not_equal = mp_not<mp_equal<T1, T2>>;

template<class T1, class T2> using mp_greater = mp_bool<(T1::value < 0 && T2::value >= 0) || ((T1::value > T2::value) && !(T1::value >= 0 && T2::value < 0))>;

template<class T1, class T2> using mp_less_equal = mp_or<mp_less<T1, T2>, mp_equal<T1, T2>>;

template<class T1, class T2> using mp_greater_equal = mp_or<mp_greater<T1, T2>, mp_equal<T1, T2>>;

namespace detail
{

#if defined( BOOST_MP11_HAS_FOLD_EXPRESSIONS ) && !BOOST_MP11_WORKAROUND( BOOST_MP11_MSVC, < 1920 )

template<class... T> struct mp_mul_impl
{
    static const auto _v = (T::value * ... * 1);
    using type = std::integral_constant<typename std::remove_const<decltype(_v)>::type, _v>;
};

#else

template<class... T> struct mp_mul_impl;

template<> struct mp_mul_impl<>
{
    using type = std::integral_constant<int, 1>;
};

#if BOOST_MP11_WORKAROUND( BOOST_MP11_GCC, < 40800 )

template<class T1, class... T> struct mp_mul_impl<T1, T...>
{
    static const decltype(T1::value * mp_mul_impl<T...>::type::value) _v = T1::value * mp_mul_impl<T...>::type::value;
    using type = std::integral_constant<typename std::remove_const<decltype(_v)>::type, _v>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class... T> struct mp_mul_impl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T...>
{
    static const
        decltype(T1::value * T2::value * T3::value * T4::value * T5::value * T6::value * T7::value * T8::value * T9::value * T10::value * mp_mul_impl<T...>::type::value)
        _v = T1::value * T2::value * T3::value * T4::value * T5::value * T6::value * T7::value * T8::value * T9::value * T10::value * mp_mul_impl<T...>::type::value;
    using type = std::integral_constant<typename std::remove_const<decltype(_v)>::type, _v>;
};

#else

template<class T1, class... T> struct mp_mul_impl<T1, T...>
{
    static const auto _v = T1::value * mp_mul_impl<T...>::type::value;
    using type = std::integral_constant<typename std::remove_const<decltype(_v)>::type, _v>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class... T> struct mp_mul_impl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T...>
{
    static const auto _v = T1::value * T2::value * T3::value * T4::value * T5::value * T6::value * T7::value * T8::value * T9::value * T10::value * mp_mul_impl<T...>::type::value;
    using type = std::integral_constant<typename std::remove_const<decltype(_v)>::type, _v>;
};

#endif

#endif

} // namespace detail

template<class... T> using mp_mul = typename detail::mp_mul_impl<T...>::type;

}
}
