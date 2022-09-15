#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11.hpp>
#include <meta_json_parser/config.h>
#include <meta_json_parser/meta_math.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/parse.cuh>
#include <meta_json_parser/json_parsers/skip.cuh>
#include <meta_json_parser/meta_utility/map_utility.h>
#include <type_traits>

struct JArrayOptions {
    struct Skip {
        struct Disable {};
        template<class StackSizeT, class SkipTypesT = JsonParsers::SkipAllTypes>
        struct Enable {
            using SkipTypes = SkipTypesT;
            using StackSize = StackSizeT;
        };
        template<int StackSizeN, class SkipTypesT = JsonParsers::SkipAllTypes>
        using Enable_c = Enable<boost::mp11::mp_int<StackSizeN>, SkipTypesT>;
        using Default = Enable_c<8, JsonParsers::SkipAllTypes>;

        template<class T>
        using GetStackSize = typename T::StackSize;

        template<class T>
        using GetSkipTypes = typename T::SkipTypes;
    };
private:
    template<class OptionsT>
    using _impl_GetArraySkip = boost::mp11::mp_map_find<OptionsT, JArrayOptions::Skip>;
public:
    template<class OptionsT>
    using GetArraySkip = boost::mp11::mp_eval_if<
        boost::mp11::mp_same<
            _impl_GetArraySkip<OptionsT>,
            void
        >,
        JArrayOptions::Skip::Default,
        boost::mp11::mp_second,
        _impl_GetArraySkip<OptionsT>
    >;
};

template<class Action, int N>
using ArrayEntry = MapEntry<boost::mp11::mp_int<N>, Action>;

using ArrayEntries_q = boost::mp11::mp_quote_trait<
    meta_json_parser::details::_impl_MapEntries
>;

template<class ...T>
using ArrayEntries = ArrayEntries_q::fn<T...>;

template<class EntriesList = boost::mp11::mp_list<>, class OptionsT = boost::mp11::mp_list<>>
struct JArray
{
    using type = JArray<EntriesList, OptionsT>;
    using Options = OptionsT;
    using Children = boost::mp11::mp_transform<
        boost::mp11::mp_second,
        EntriesList
    >;
    using Indices = boost::mp11::mp_transform<
        boost::mp11::mp_first,
        EntriesList
    >;
    using UniqueIndices = boost::mp11::mp_equal<
        boost::mp11::mp_size<Indices>,
        boost::mp11::mp_size<boost::mp11::mp_unique<Indices>>
    >;
    static_assert(UniqueIndices::value, "Indices must be unique in JArray.");
    using SortedEntries = boost::mp11::mp_sort_q<
        EntriesList,
        boost::mp11::mp_bind_q<
            boost::mp11::mp_quote<boost::mp11::mp_less>,
            boost::mp11::mp_bind<boost::mp11::mp_first, boost::mp11::_1>,
            boost::mp11::mp_bind<boost::mp11::mp_first, boost::mp11::_2>
        >
    >;
    using SortedIndices = boost::mp11::mp_transform<
        boost::mp11::mp_first,
        SortedEntries
    >;
    using MaxIndex = boost::mp11::mp_eval_or<
        boost::mp11::mp_int<-1>,
        boost::mp11::mp_back,
        SortedIndices
    >;
    using SkipOption = JArrayOptions::GetArraySkip<Options>;

    template<class Idx>
    using IndexPresent = boost::mp11::mp_not<std::is_same<
        boost::mp11::mp_map_find<
            SortedEntries,
            Idx
        >,
        void
    >>;

    using SkipStackSize = boost::mp11::mp_eval_or<
        boost::mp11::mp_int<0>,
        JArrayOptions::Skip::GetStackSize,
        SkipOption
    >;

    using SkipTypes = boost::mp11::mp_eval_or<
        boost::mp11::mp_list<>,
        JArrayOptions::Skip::GetSkipTypes,
        SkipOption
    >;

    using SkippingDisabled = std::is_same<
        SkipOption,
        JArrayOptions::Skip::Disable
    >;

    static constexpr bool SkippingDisabled_v = SkippingDisabled::value;

    template<class Idx>
    static constexpr bool IndexPresent_v = IndexPresent<Idx>::value;

    using MemoryRequests = boost::mp11::mp_eval_if<
        SkippingDisabled,
        boost::mp11::mp_list<>,
        JsonParsers::SkipRequests,
        SkipStackSize
    >;

    template<class Idx, class KernelContextT>
    static __device__ INLINE_METHOD typename std::enable_if_t<
        !IndexPresent_v<Idx> && SkippingDisabled_v,
        ParsingError
    > DispatchIndex(KernelContextT& kc)
    {
        static_assert(!SkippingDisabled_v, "Skipping indices is disabled.");
        return ParsingError::Other;
    }

    template<class Idx, class KernelContextT>
    static __device__ INLINE_METHOD typename std::enable_if_t<
        !IndexPresent_v<Idx> && !SkippingDisabled_v,
        ParsingError
    > DispatchIndex(KernelContextT& kc)
    {
        return JsonParsers::Skip<KernelContextT, SkipStackSize, SkipTypes>(kc);
    }

    template<class Idx, class KernelContextT>
    static __device__ INLINE_METHOD typename std::enable_if_t<
        IndexPresent_v<Idx>,
        ParsingError
    > DispatchIndex(KernelContextT& kc)
    {
        using Action = boost::mp11::mp_second<boost::mp11::mp_map_find<
            SortedEntries,
            Idx
        >>;
        return Action::Invoke(kc);
    }

    template<class SkippingDisabledT, class KernelContextT>
    static __device__ INLINE_METHOD typename std::enable_if_t<
        SkippingDisabledT::value,
        ParsingError
    > DispatchSkipping(KernelContextT& kc) {
        return ParsingError::Other;
    }

    template<class SkippingDisabledT, class KernelContextT>
    static __device__ INLINE_METHOD typename std::enable_if_t<
        !SkippingDisabledT::value,
        ParsingError
    > DispatchSkipping(KernelContextT& kc) {
        return JsonParsers::Skip<KernelContextT, SkipStackSize, SkipTypes>(kc);
    }

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WS = typename RT::WorkGroupSize;
        if (kc.wgr.PeekChar(0) != '[')
            return ParsingError::Other;
        kc.wgr.AdvanceBy(1);
        ParsingError err = ParsingError::None;
        err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
        if (err != ParsingError::None)
            return err;
        char c = kc.wgr.PeekChar(0);
        if (c == ']')
        {
            kc.wgr.AdvanceBy(1);
            return ParsingError::None;
        }
        bool endOfArray = false;
        //TODO add mis-wut/mp11 as submodule
        boost::mp11::mp_for_each<boost::mp11::mp_iota_c<MaxIndex::value + 1>>([&](auto i)
        {
            using Idx = boost::mp11::mp_int<decltype(i)::value>;
            if (err != ParsingError::None || endOfArray)
                return;
            err = DispatchIndex<Idx>(kc);
            if (err != ParsingError::None)
                return;
            err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
            if (err != ParsingError::None)
                return;
            c = kc.wgr.PeekChar(0);
            kc.wgr.AdvanceBy(1);
            if (c == ',')
            {
                err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
                if (err != ParsingError::None)
                    return;
            }
            else if (c == ']')
            {
                endOfArray = true;
                return;
            }
            else
            {
                err = ParsingError::Other;
                return;
            }
        });
        if (err != ParsingError::None)
            return err;
        if (endOfArray) {
            return ParsingError::None;
        }
        while (kc.wgr.PeekChar(0) != '\0')
        {
            err = DispatchSkipping<SkippingDisabled>(kc);
            if (err != ParsingError::None)
                return err;
            err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
            if (err != ParsingError::None)
                return err;
            c = kc.wgr.PeekChar(0);
            if (c == ',')
            {
                kc.wgr.AdvanceBy(1);
                err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
                if (err != ParsingError::None)
                    return err;
            }
            else if (c == ']') {
                kc.wgr.AdvanceBy(1);
                break;
            }
            else
                return ParsingError::Other;
        }
        return ParsingError::None;
    }
};

