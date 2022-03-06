#ifndef META_JSON_PARSER_METASTRING_H
#define META_JSON_PARSER_METASTRING_H
#include <boost/mp11/list.hpp>
#include <boost/mp11/integral.hpp>
#include <typestring/typstring.hh>

namespace meta_json_parser::details {
    template<typename T>
    struct _impl_typestring_to_metastring {
        using type = boost::mp11::mp_list<>;
    };

    template<char ...Chars>
    struct _impl_typestring_to_metastring<irqus::typestring<Chars...>> {
        using type = boost::mp11::mp_list<boost::mp11::mp_int<Chars>...>;
    };
}

template<class T>
using typestring_to_metastring = typename meta_json_parser::details::_impl_typestring_to_metastring<T>::type;

#ifndef metastring
#define metastring(STR) typestring_to_metastring<typestring_is(STR)>
#endif

#endif //META_JSON_PARSER_METASTRING_H
