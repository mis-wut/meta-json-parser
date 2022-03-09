#ifndef META_JSON_PARSER_JREALNUMBER_CUH
#define META_JSON_PARSER_JREALNUMBER_CUH
#include <utility>
#include <boost/mp11/list.hpp>
#include <boost/mp11/map.hpp>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parsers/float.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <type_traits>

struct JRealNumberOptions {
    struct JRealNumberTransformer {
        struct DefaultRealNumberTransformer {
            template<class T>
            inline __device__ T operator()(T c) const { return c; }
        };
    };

    struct JRealNumberSign {
        using Signed = std::true_type;
        using Unsigned = std::false_type;
        using DefaultSign = Signed;
    };

    struct JRealNumberExponent {
        using WithoutExponent = std::false_type;
        using WithExponent = std::true_type;
        using DefaultExponent = WithExponent;
    };

private:
    template<class OptionsT>
    using _impl_GetRealNumberTransformer = boost::mp11::mp_map_find<OptionsT, JRealNumberOptions::JRealNumberTransformer>;

    template<class OptionsT>
    using _impl_GetRealNumberSign = boost::mp11::mp_map_find<OptionsT, JRealNumberOptions::JRealNumberSign>;


    template<class OptionsT>
    using _impl_GetRealNumberExponent = boost::mp11::mp_map_find<OptionsT, JRealNumberOptions::JRealNumberExponent>;
public:
    template<class OptionsT>
    using GetRealNumberTransformer = boost::mp11::mp_eval_if<
            boost::mp11::mp_same<
                    _impl_GetRealNumberTransformer<OptionsT>,
                    void
            >,
            JRealNumberOptions::JRealNumberTransformer::DefaultRealNumberTransformer,
            boost::mp11::mp_second,
            _impl_GetRealNumberTransformer<OptionsT>
    >;

    template<class OptionsT>
    using GetRealNumberSign = boost::mp11::mp_eval_if<
            boost::mp11::mp_same<
                    _impl_GetRealNumberSign<OptionsT>,
                    void
            >,
            JRealNumberOptions::JRealNumberSign::DefaultSign,
            boost::mp11::mp_second,
            _impl_GetRealNumberSign<OptionsT>
    >;

    template<class OptionsT>
    using GetRealNumberExponent = boost::mp11::mp_eval_if<
            boost::mp11::mp_same<
                    _impl_GetRealNumberExponent<OptionsT>,
                    void
            >,
            JRealNumberOptions::JRealNumberExponent::DefaultExponent,
            boost::mp11::mp_second,
            _impl_GetRealNumberExponent<OptionsT>
    >;
};

template<class OutT, class TagT, class OptionsT = boost::mp11::mp_list<>>
struct JRealNumber
{
    using type = JRealNumber<OutT, TagT, OptionsT>;
    using ParsingType = OutT;
    using Options = OptionsT;
    using NumberTransformer = JRealNumberOptions::GetRealNumberTransformer<Options>;
    using NumberSign = JRealNumberOptions::GetRealNumberSign<Options>;
    using NumberExponent = JRealNumberOptions::GetRealNumberExponent<Options>;
    using Out = decltype(std::declval<NumberTransformer>()(std::declval<ParsingType>()));
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, Out>>;
    using MemoryRequests = JsonParsers::FloatRequests<ParsingType>;
    static_assert(std::is_floating_point_v<ParsingType>, "ParsingType must be floating point.");

#ifdef HAVE_LIBCUDF
    using CudfColumnConverter = CudfNumericColumn<JRealNumber, OutT>;
#endif

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        NumberTransformer transformer;
        return JsonParsers::Float<ParsingType, NumberSign, NumberExponent>(kc, [&](auto&& result) {
            kc.om.template Get<KernelContextT, TagT>() = transformer(result);
        });
    }
};

#endif //META_JSON_PARSER_JREALNUMBER_CUH
