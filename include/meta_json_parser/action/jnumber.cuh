#pragma once
#include <utility>
#include <boost/mp11/list.hpp>
#include <boost/mp11/map.hpp>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <type_traits>

struct JNumberOptions {
    struct JNumberTransformer {
        struct DefaultNumberTransformer {
            template<class T>
            inline __device__ T operator()(T c) const { return c; }
        };
    };

    struct JNumberSign {
        using Signed = std::true_type;
        using Unsigned = std::false_type;
        using DefaultSign = Unsigned;
    };

private:
    template<class OptionsT>
    using _impl_GetNumberTransformer = boost::mp11::mp_map_find<OptionsT, JNumberOptions::JNumberTransformer>;

    template<class OptionsT>
    using _impl_GetNumberSign = boost::mp11::mp_map_find<OptionsT, JNumberOptions::JNumberSign>;
public:
    template<class OptionsT>
    using GetNumberTransformer = boost::mp11::mp_eval_if<
            boost::mp11::mp_same<
                    _impl_GetNumberTransformer<OptionsT>,
                    void
            >,
            JNumberOptions::JNumberTransformer::DefaultNumberTransformer,
            boost::mp11::mp_second,
            _impl_GetNumberTransformer<OptionsT>
    >;

    template<class OptionsT>
    using GetNumberSign = boost::mp11::mp_eval_if<
            boost::mp11::mp_same<
                    _impl_GetNumberSign<OptionsT>,
                    void
            >,
            JNumberOptions::JNumberSign::DefaultSign,
            boost::mp11::mp_second,
            _impl_GetNumberSign<OptionsT>
    >;
};

template<class OutT, class TagT, class OptionsT = boost::mp11::mp_list<>>
struct JNumber
{
    using type = JNumber<OutT, TagT, OptionsT>;
    using ParsingType = OutT;
    using Options = OptionsT;
    using NumberTransformer = JNumberOptions::GetNumberTransformer<Options>;
    using NumberSign = JNumberOptions::GetNumberSign<Options>;
    using Out = decltype(std::declval<NumberTransformer>()(std::declval<ParsingType>()));
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, Out>>;
    using MemoryRequests = JsonParse::UnsignedIntegerRequests<ParsingType>;
    static_assert(std::is_integral_v<ParsingType>, "OutT must be integral.");

#ifdef HAVE_LIBCUDF
    using CudfColumnConverter = CudfNumericColumn<JNumber, OutT>;
#endif

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        NumberTransformer transformer;
        return JsonParse::Integer<NumberSign, ParsingType>(kc, [&](auto&& result) {
            kc.om.template Get<KernelContextT, TagT>() = transformer(result);
        });
    }
};
