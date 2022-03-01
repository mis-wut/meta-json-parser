#ifndef HASH_PY_LENGTH_REPRESENTATION_H
#define HASH_PY_LENGTH_REPRESENTATION_H
#include <boost/mp11/integral.hpp>
#include <boost/mp11/utility.hpp>

namespace meta_json_parser::details {
    template<class ValueT>
    struct _impl_LengthRepresentation {
        using ValueType = typename ValueT::value_type;
        static constexpr ValueType Value = ValueT::value;
        static constexpr bool IsSigned = Value < 0;
        static constexpr ValueType NextValue = IsSigned ? -(Value / 10) : (Value / 10);
        using NextValueT = std::integral_constant<ValueType, NextValue>;

        using type = boost::mp11::mp_int<
            1 + (IsSigned ? 1 : 0) + boost::mp11::mp_eval_if_c<
                (NextValue == 0),
                std::integral_constant<ValueType, 0>,
                boost::mp11::mp_quote_trait<meta_json_parser::details::_impl_LengthRepresentation>::fn,
                NextValueT
            >::value
        >;
    };
}

template<class ValueT>
using LengthRepresentation = typename meta_json_parser::details::_impl_LengthRepresentation<ValueT>::type;


template<uint64_t Value>
using LengthRepresentation_c = LengthRepresentation<std::integral_constant<uint64_t, Value>>;

#endif //HASH_PY_LENGTH_REPRESENTATION_H
