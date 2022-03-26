#ifndef META_JSON_PARSER_JDATETIME_CUH
#define META_JSON_PARSER_JDATETIME_CUH
#include <utility>
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/output_printer.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/json_parsers/datetime.cuh>
#include <meta_json_parser/json_parsers/datetime_token_parser.h>
#include <meta_json_parser/action/datetime/datetime_options.h>

// TODO: check if this is needed
#ifdef HAVE_LIBCUDF
#include <boost/mp11/utility.hpp>
#endif

template<class TokensT, class OutTypeT, class TagT, class OptionsT = boost::mp11::mp_list<>>
struct JDatetimeToken {
    using type = JDatetimeToken<TokensT, OutTypeT, TagT, OptionsT>;

    using Tokens = TokensT;
    using Options = OptionsT;
    using OutType = OutTypeT;
    using TimestampType = JDatetimeOptions::GetTimestampResolution<Options>;
    using DatetimeTransformer = JDatetimeOptions::GetDatetimeTransformer<Options>;
    using Out = decltype (std::declval<DatetimeTransformer>()(std::declval<OutType>()));
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, Out>>;
    using MemoryRequests = JsonParsers::DatetimeRequests;

#ifdef HAVE_LIBCUDF
    // NOTE: use Out rather than OutType or OutTypeT
    using CudfColumnConverter = CudfDatetimeColumn<JDatetimeToken, Out, TimestampType>;
#endif

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        using ParseMilliseconds = std::is_same<TimestampType, JDatetimeOptions::TimestampResolution::Milliseconds>;
        DatetimeTransformer transformer;
        return JsonParsers::Datetime<ParseMilliseconds, Tokens, OutType>(kc, [&](auto&& result) {
            kc.om.template Get<KernelContextT, TagT>() = transformer(result);
        });
    }
};

template<class MetaString, class OutTypeT, class TagT, class OptionsT = boost::mp11::mp_list<>>
struct JDatetime : public JDatetimeToken<JsonParsers::DatetimeTokenParser<MetaString>, OutTypeT, TagT, OptionsT> {
};

#endif //META_JSON_PARSER_JDATETIME_CUH
