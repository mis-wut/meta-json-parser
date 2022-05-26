#ifndef META_JSON_PARSER_SKIP_ACTION_CUH
#define META_JSON_PARSER_SKIP_ACTION_CUH
#include <boost/mp11.hpp>
#include <meta_json_parser/json_parsers/skip.cuh>

template<class StackSizeT, class SkipTypesT = JsonParsers::SkipAllTypes>
struct SkipAction
{
    using type = SkipAction<StackSizeT, SkipTypesT>;
    using StackSize = StackSizeT;
    using SkipTypes = SkipTypesT;
    using MemoryRequests = JsonParsers::SkipRequests<StackSize>;

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        return JsonParsers::Skip<KernelContextT, StackSize, SkipTypes>(kc);
    }
};

template<int StackSizeN, class SkipTypesT = JsonParsers::SkipAllTypes>
using SkipAction_c = SkipAction<boost::mp11::mp_int<StackSizeN>, SkipTypesT>;

#endif //META_JSON_PARSER_SKIP_ACTION_CUH
