#ifndef META_JSON_PARSER_JDATETIME_CUH
#define META_JSON_PARSER_JDATETIME_CUH
#include <utility>
#include <meta_json_parser/json_parsers/datetime.cuh>
#include <meta_json_parser/json_parsers/datetime_token_parser.h>

struct JDatetimeOptions {
    struct JDatetimeTransformer {
        struct DefaultDatetimeTransformer {
            template<class T>
            inline __device__ T operator()(T c) const { return c; }
        };
    };

    struct TimestampResolution {
        using Seconds = uint32_t;
        using Milliseconds = uint64_t;
        using Default = Seconds;
    };

private:
    template<class OptionsT>
    using _impl_GetDatetimeTransformer = boost::mp11::mp_map_find<OptionsT, JDatetimeOptions::JDatetimeTransformer>;

    template<class OptionsT>
    using _impl_GetTimestampResolution = boost::mp11::mp_map_find<OptionsT, JDatetimeOptions::TimestampResolution>;
public:
    template<class OptionsT>
    using GetDatetimeTransformer = boost::mp11::mp_eval_if<
        boost::mp11::mp_same<
                _impl_GetDatetimeTransformer<OptionsT>,
        void
        >,
        JDatetimeOptions::JDatetimeTransformer::DefaultDatetimeTransformer,
        boost::mp11::mp_second,
        _impl_GetDatetimeTransformer<OptionsT>
    >;

    template<class OptionsT>
    using GetTimestampResolution = boost::mp11::mp_eval_if<
        boost::mp11::mp_same<
            _impl_GetTimestampResolution<OptionsT>,
            void
        >,
        JDatetimeOptions::TimestampResolution::Default,
        boost::mp11::mp_second,
        _impl_GetTimestampResolution<OptionsT>
    >;
};

template<class TokensT, class TagT, class OptionsT = boost::mp11::mp_list<>>
struct JDatetimeToken {
    using type = JDatetimeToken<TokensT, TagT, OptionsT>;

    using Tokens = TokensT;
    using Options = OptionsT;
    using TimestampType = JDatetimeOptions::GetTimestampResolution<Options>;
    using DatetimeTransformer = JDatetimeOptions::GetDatetimeTransformer<Options>;
    using Out = decltype (std::declval<DatetimeTransformer>()(std::declval<TimestampType>()));
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, Out>>;
    using MemoryRequests = JsonParsers::DatetimeRequests;

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        using Milliseconds = std::is_same<TimestampType, JDatetimeOptions::TimestampResolution::Milliseconds>;
        DatetimeTransformer transformer;
        return JsonParsers::Datetime<Milliseconds, Tokens>(kc, [&](auto&& result) {
            kc.om.template Get<KernelContextT, TagT>() = transformer(result);
        });
    }
};

template<class MetaString, class TagT, class OptionsT = boost::mp11::mp_list<>>
struct JDatetime : public JDatetimeToken<JsonParsers::DatetimeTokenParser<MetaString>, TagT, OptionsT> {
};

#endif //META_JSON_PARSER_JDATETIME_CUH
