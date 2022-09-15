#ifndef META_JSON_PARSER_MAP_UTILITY_H
#define META_JSON_PARSER_MAP_UTILITY_H

template<class Key, class Action>
using MapEntry = boost::mp11::mp_list<
    Key,
    Action
>;

namespace meta_json_parser::details {
    template<class ...T>
    struct _impl_MapEntries {
        using List = boost::mp11::mp_list<T...>;
        using Size = boost::mp11::mp_size<List>;
        static_assert(Size::value % 2 == 0, "Number of elements passed to MapEntries must be even.");
        using ResultOrder = boost::mp11::mp_iota_c<Size::value / 2>;

        template<class ListKey>
        struct MapEntryConstructor_trait {
            using Key = boost::mp11::mp_at_c<List, ListKey::value * 2>;
            using Action = boost::mp11::mp_at_c<List, ListKey::value * 2 + 1>;
            using type = MapEntry<Key, Action>;
        };

        using type = boost::mp11::mp_transform_q<
            boost::mp11::mp_quote_trait<MapEntryConstructor_trait>,
            ResultOrder
        >;
    };
}

using MapEntries_q = boost::mp11::mp_quote_trait<
    meta_json_parser::details::_impl_MapEntries
>;

template<class ...T>
using MapEntries = MapEntries_q::fn<T...>;

#endif //META_JSON_PARSER_MAP_UTILITY_H
