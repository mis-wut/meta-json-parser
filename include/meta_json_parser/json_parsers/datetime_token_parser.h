#ifndef META_JSON_PARSER_DATETIME_TOKEN_PARSER_H
#define META_JSON_PARSER_DATETIME_TOKEN_PARSER_H
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/map.hpp>
#include <meta_json_parser/meta_utility/safe_drop.h>
#include "datetime.cuh"

namespace JsonParsers {
    namespace details {
        using MetaChar = boost::mp11::mp_int<'%'>;

        using KnownSymbols = boost::mp11::mp_list<
            boost::mp11::mp_list<
                boost::mp11::mp_int<'Y'>, JsonParsers::DatetimeTokens::YearDigit_c<4>
            >,
            boost::mp11::mp_list<
                boost::mp11::mp_int<'m'>, JsonParsers::DatetimeTokens::MonthDigit_c<2>
            >,
            boost::mp11::mp_list<
                boost::mp11::mp_int<'d'>, JsonParsers::DatetimeTokens::DayDigit_c<2>
            >,
            boost::mp11::mp_list<
                boost::mp11::mp_int<'H'>, JsonParsers::DatetimeTokens::HourDigit_c<2>
            >,
            boost::mp11::mp_list<
                boost::mp11::mp_int<'M'>, JsonParsers::DatetimeTokens::MinuteDigit_c<2>
            >,
            boost::mp11::mp_list<
                boost::mp11::mp_int<'S'>, JsonParsers::DatetimeTokens::SecondDigit_c<2>
            >
        >;

        template<class MetaString>
        struct _impl_MetaCharParser {
            using Length = boost::mp11::mp_size<MetaString>;
            static_assert(Length::value > 1, "MetaChar cannot be the last char in the string.");
            using NextChar = boost::mp11::mp_eval_if_c<
                (Length::value <= 1),
                boost::mp11::mp_int<'\0'>,
                boost::mp11::mp_second,
                MetaString
            >;
            using TokenFound = boost::mp11::mp_map_find<KnownSymbols, NextChar>;
            static_assert(!std::is_same<TokenFound, void>::value, "Unknown symbol after MetaChar.");

            using Token = boost::mp11::mp_eval_if<
                std::is_same<TokenFound, void>,
                void,
                boost::mp11::mp_second,
                TokenFound
            >;
            using Rest = safe_drop_c<MetaString, 2>;
        };

        template<class MetaString>
        struct _impl_TextParser {
            using Length = boost::mp11::mp_size<MetaString>;
            using NextMeta = boost::mp11::mp_find<MetaString, MetaChar>;
            using LocalText = boost::mp11::mp_take<MetaString, NextMeta>;
            using AfterMeta = boost::mp11::mp_eval_if_c<
                (NextMeta::value + 1 >= Length::value),
                boost::mp11::mp_int<MetaChar::value + 1>, // Not meta
                boost::mp11::mp_at,
                MetaString,
                boost::mp11::mp_int<NextMeta::value + 1>
            >;
            using EscapedMeta = boost::mp11::mp_bool<AfterMeta::value == MetaChar::value>;
            using SubParser = boost::mp11::mp_eval_if_not<
                EscapedMeta,
                _impl_TextParser<boost::mp11::mp_list<>>,
                _impl_TextParser,
                safe_drop_c<MetaString, NextMeta::value + 2>
            >;
            using SubText = boost::mp11::mp_eval_if_not<
                EscapedMeta,
                typename SubParser::Text,
                boost::mp11::mp_append,
                boost::mp11::mp_list<MetaChar>,
                typename SubParser::Text
            >;
            using LocalRest = safe_drop<MetaString, NextMeta>;
            using Text = boost::mp11::mp_remove<
                boost::mp11::mp_append<LocalText, SubText>,
                boost::mp11::mp_int<'\0'>
            >;

            using Token = JsonParsers::DatetimeTokens::Text<Text>;
            using Rest = boost::mp11::mp_if<
                EscapedMeta,
                typename SubParser::Rest,
                LocalRest
            >;
        };

        template<>
        struct _impl_TextParser<boost::mp11::mp_list<>> {
            using Text = boost::mp11::mp_list<>;
            using Rest = boost::mp11::mp_list<>;
        };

        template<class MetaString>
        struct _impl_DatetimeTokenParser {
            using Meta = boost::mp11::mp_bool<boost::mp11::mp_first<MetaString>::value == MetaChar::value>;
            using MetaCharParser = boost::mp11::mp_eval_if_not<
                Meta,
                void,
                _impl_MetaCharParser,
                MetaString
            >;
            using TextParser = boost::mp11::mp_eval_if<
                Meta,
                void,
                _impl_TextParser,
                MetaString
            >;
            using InnerParser = boost::mp11::mp_if<
                Meta,
                MetaCharParser,
                TextParser
            >;
            using Tokens = boost::mp11::mp_push_front<
                typename _impl_DatetimeTokenParser<typename InnerParser::Rest>::Tokens,
                typename InnerParser::Token
            >;
        };

        template<>
        struct _impl_DatetimeTokenParser<boost::mp11::mp_list<>> {
            using Tokens = boost::mp11::mp_list<>;
        };
    }

    template<class MetaString>
    using DatetimeTokenParser = typename JsonParsers::details::_impl_DatetimeTokenParser<MetaString>::Tokens;
}

#endif //META_JSON_PARSER_DATETIME_TOKEN_PARSER_H
