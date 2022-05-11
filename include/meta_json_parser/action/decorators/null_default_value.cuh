#ifndef META_JSON_PARSER_NULL_DEFAULT_VALUE_CUH
#define META_JSON_PARSER_NULL_DEFAULT_VALUE_CUH
#include <type_traits>
#include <boost/mp11.hpp>
#include <meta_json_parser/action_iterator.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/action/jstring.cuh>

/**
 * Implementation of constant default value used by NullDefaultValue.
 * @tparam ConstantT Constant type, must implement constexpr static ConstantT::value (ex. mp_int).
 */
template<class ConstantT>
struct ConstantDefaultValue {
    using type = ConstantDefaultValue<ConstantT>;
    using Constant = ConstantT;
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) {
        kc.om.template Get<KernelContextT, TagT>() = Constant::value;
    }
};

struct SkipDefaultValue {
    using type = SkipDefaultValue;
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) { }
};

template<class BytesT>
struct EmptyStaticStringDefaultValue {
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using Bytes = BytesT;
        char (&result)[Bytes::value] = kc.om.template Get<KernelContextT, TagT>().template Alias<char[Bytes::value]>();
        uint32_t offset = 0;
        while (offset < Bytes::value) {
            uint32_t worker_offset = offset + RT::WorkerId();
            if (worker_offset < Bytes::value)
                result[worker_offset] = '\0';
            offset += RT::GroupSize();
        }
    }
};

struct EmptyStringDefaultValueStatic {
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using BaseAction = typename KC::BaseAction;
        using Action = GetTaggedAction<BaseAction, TagT>;
        using Bytes = typename Action::BytesT;
        char (&result)[Bytes::value] = kc.om.template Get<KernelContextT, TagT>().template Alias<char[Bytes::value]>();
        uint32_t offset = 0;
        while (offset < Bytes::value) {
            uint32_t worker_offset = offset + RT::WorkerId();
            if (worker_offset < Bytes::value)
                result[worker_offset] = '\0';
            offset += RT::GroupSize();
        }
    }
};

struct EmptyStringDefaultValueDynamic1 {
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using BaseAction = typename KC::BaseAction;
        using Action = GetTaggedAction<BaseAction, TagT>;
        using DynamicStringRequestTag = typename Action::DynamicStringRequestTag;
        using LengthRequestTag = typename Action::LengthRequestTag;

        const uint32_t max_offset = kc.om.template DynamicSize<KernelContextT, DynamicStringRequestTag>();
        char* result = kc.om.template Pointer<KernelContextT, DynamicStringRequestTag>();

        kc.om.template Get<KernelContextT, LengthRequestTag>() = 0; // Replace with 1 if empty string requires null byte

        uint32_t offset = 0;
        while (offset < max_offset)
        {
            uint32_t worker_offset = offset + RT::WorkerId();
            if (worker_offset < max_offset)
                result[worker_offset] = '\0';
            offset += RT::GroupSize();
        }
    }
};

struct EmptyStringDefaultValueDynamic2 {
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using BaseAction = typename KC::BaseAction;
        using Action = GetTaggedAction<BaseAction, TagT>;
        using DynamicStringRequestTag = typename Action::DynamicStringRequestTag;
        using OffsetsRequestTag = typename Action::OffsetsRequestTag;
        using LengthRequestTag = typename Action::LengthRequestTag;

        const uint32_t max_offset = kc.om.template DynamicSize<KernelContextT, DynamicStringRequestTag>();
        kc.om.template Get<KernelContextT, OffsetsRequestTag>() = kc.wgr.GroupDistance() + 1; // +1 to skip "
        kc.om.template Get<KernelContextT, LengthRequestTag>() = 0; // Replace with 1 if empty string requires null byte
    }
};

struct EmptyStringDefaultValueDynamic3 {
    template<class KernelContextT, class TagT>
    static void __device__ __forceinline__ StoreValue(KernelContextT& kc) {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using BaseAction = typename KC::BaseAction;
        using Action = GetTaggedAction<BaseAction, TagT>;
        using DynamicStringInternalRequestTag = typename Action::DynamicStringInternalRequestTag;
        using LengthRequestTag = typename Action::LengthRequestTag;

        char* result = kc.om.template Pointer<KernelContextT, DynamicStringInternalRequestTag>();
        const uint32_t max_offset = kc.om.template DynamicSize<KernelContextT, DynamicStringInternalRequestTag>();

        kc.om.template Get<KernelContextT, LengthRequestTag>() = 0; // Replace with 1 if empty string requires null byte

        uint32_t offset = 0;
        while (offset < max_offset)
        {
            uint32_t worker_offset = offset + RT::WorkerId();
            if (worker_offset < max_offset)
                result[worker_offset] = '\0';
            offset += RT::GroupSize();
        }
    }
};

/**
 * Decorator that return default values if a field json is `null`
 * @tparam ActionT Action to decorate.
 * @tparam DefaultMapT MP11 map that maps tags to default values that will be used by decorator. Each tag in
 * OutputRequests should be covered in the map. DefaultValue type have to implement
 * StoreValue<KernelContext, Tag>(KernelContext& kc) that should save default value in OutputManager in a way that
 * matches OutputRequests of an action.
 * Mp11Map[Tag] -> DefaultValue
 */
template<class ActionT, class DefaultMapT>
struct NullDefaultValue : public ActionT {
    using type = NullDefaultValue<ActionT, DefaultMapT>;
    using Action = ActionT;
    using DefaultMap = DefaultMapT;

private:
    using ActionOutputRequests = typename Action::OutputRequests;
    using ActionOutputTags = boost::mp11::mp_transform<
        boost::mp11::mp_first,
        ActionOutputRequests
    >;
    using MapTags = boost::mp11::mp_transform<
        boost::mp11::mp_first,
        DefaultMap
    >;
    using UnionTags = boost::mp11::mp_set_union<ActionOutputTags, MapTags>;
    using IntersectionTags = boost::mp11::mp_set_intersection<ActionOutputTags, MapTags>;
    using AreSetsTheSame = boost::mp11::mp_bool<
        (boost::mp11::mp_size<UnionTags>::value == boost::mp11::mp_size<IntersectionTags>::value)
    >;
    static_assert(AreSetsTheSame::value, "Keys in DefaultMapT should match tags in ActionT::RequestsOutputs.");
public:
    using MemoryRequests = boost::mp11::mp_append<typename ActionT::MemoryRequests, JsonParse::IsNullRequests>;

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc) {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        ParsingError err = ParsingError::None;
        ParsingError err2 = JsonParse::IsNull(kc, [&](bool result) {
            if (result) {
                // fill with defaults
                boost::mp11::mp_for_each<MapTags>([&](auto tag){
                    using Tag = decltype(tag);
                    using MapEntry = boost::mp11::mp_map_find<DefaultMap, Tag>;
                    using DefaultValue = boost::mp11::mp_second<MapEntry>;
                    DefaultValue::template StoreValue<KernelContextT, Tag>(kc);
                });
            } else {
                err = Action::Invoke(kc);
            }
        });
        if (err != ParsingError::None)
            return err;
        if (err2 != ParsingError::None)
            return err2;
        return ParsingError::None;
    }

};

template<class ActionT, class IntegerT>
struct NullDefaultInteger : public NullDefaultValue <
    ActionT,
    boost::mp11::mp_list<
        boost::mp11::mp_list<
            typename ActionT::Tag,
            ConstantDefaultValue<IntegerT>
        >
    >
> {};

template<class ActionT>
struct NullDefaultEmptyString {
    // Always fails
    static_assert(
        std::is_same_v<ActionT, boost::mp11::mp_list<ActionT>>,
        "Unsupported action in NullDefaultEmptyString"
    );
};

template<class BytesT, class TagT, class OptionsT>
struct NullDefaultEmptyString<JStringStaticCopy<BytesT, TagT, OptionsT>> : public NullDefaultValue<
    JStringStaticCopy<BytesT, TagT, OptionsT>,
    boost::mp11::mp_list<
        boost::mp11::mp_list<
            TagT,
            EmptyStaticStringDefaultValue<BytesT>
        >
    >
> {};

template<class TagT, class OptionsT>
struct NullDefaultEmptyString<JStringDynamicCopy<TagT, OptionsT>> : public NullDefaultValue<
    JStringDynamicCopy<TagT, OptionsT>,
    boost::mp11::mp_list<
        boost::mp11::mp_list<
            typename JStringDynamicCopy<TagT, OptionsT>::LengthRequestTag,
            ConstantDefaultValue<boost::mp11::mp_int<0>>
        >,
        boost::mp11::mp_list<
            typename JStringDynamicCopy<TagT, OptionsT>::DynamicStringRequestTag,
            SkipDefaultValue
        >
    >
> {};

template<class TagT, class OptionsT>
struct NullDefaultEmptyString<JStringDynamicCopyV3<TagT, OptionsT>> : public NullDefaultValue<
    JStringDynamicCopyV3<TagT, OptionsT>,
    boost::mp11::mp_list<
        boost::mp11::mp_list<
            typename JStringDynamicCopyV3<TagT, OptionsT>::DynamicStringInternalRequestTag,
            ConstantDefaultValue<boost::mp11::mp_int<0>>
        >,
        boost::mp11::mp_list<
            typename JStringDynamicCopyV3<TagT, OptionsT>::DynamicStringRequestTag,
            SkipDefaultValue
        >
    >
> {};

#endif //META_JSON_PARSER_NULL_DEFAULT_VALUE_CUH
