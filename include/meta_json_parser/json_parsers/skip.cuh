#ifndef META_JSON_PARSER_SKIP_CUH
#define META_JSON_PARSER_SKIP_CUH
#include <meta_json_parser/byte_algorithms.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/json_parsers/float.cuh>
#include <meta_json_parser/json_parsers/stack_token.cuh>

namespace JsonParsers {
    struct SkipJsonTypes {
        struct Object{};
        struct Array{};
        struct Number{};
        struct String{};
        struct Boolean{};
        struct Null{};
    };

    using SkipAllTypes = boost::mp11::mp_list<
        SkipJsonTypes::Object,
        SkipJsonTypes::Array,
        SkipJsonTypes::Number,
        SkipJsonTypes::String,
        SkipJsonTypes::Boolean,
        SkipJsonTypes::Null
    >;

    template<class TypesToSkipT>
    struct CheckForHelper {
        static constexpr bool Comma = boost::mp11::mp_or<
            boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Object>,
            boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Array>
        >::value;

        static constexpr bool Braces = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Object>::value;

        static constexpr bool Brackets = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Array>::value;

        static constexpr bool Quotes = boost::mp11::mp_or<
            boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Object>,
            boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::String>
        >::value;

        static constexpr bool Null = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Null>::value;

        static constexpr bool Boolean = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Boolean>::value;

        static constexpr bool Number = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Number>::value;

        static constexpr bool Object = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Object>::value;
        static constexpr bool Array = boost::mp11::mp_set_contains<TypesToSkipT, SkipJsonTypes::Array>::value;
    };

    template<class StackSizeT>
    using StackRequest = MemoryRequest_c<
        StackSizeT::value * sizeof(StackToken),
        MemoryUsage::ActionUsage
    >;

    template<class StackSizeT>
    using SkipRequests = boost::mp11::mp_append<
        JsonParsers::FloatRequests<float>,
        JsonParse::BooleanRequests,
        boost::mp11::mp_list<StackRequest<StackSizeT>>
    >;

    /**
     * Skips up to StackSizeT nested JSON objects.
     * @tparam KernelContextT
     * @tparam StackSizeT Stack sized used to keep track of nested objects. Ex. mp_int<N>
     */
    template<class KernelContextT, class StackSizeT, class TypesToSkipT = SkipAllTypes>
    __device__ INLINE_METHOD ParsingError Skip(KernelContextT& _kc) {
        using StackSize = StackSizeT;
        using TypesToSkip = TypesToSkipT;
        using CheckFor = CheckForHelper<TypesToSkip>;
        constexpr int STACK_SIZE = StackSize::value;
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WS = typename RT::WorkGroupSize;

        // TODO no stack if NOT (Array OR Object) IN TypesToSkip
        StackToken (&stack)[STACK_SIZE] = _kc.m3
            .template Receive<StackRequest<StackSize>>()
            .template Alias<StackToken[StackSize::value]>();

        ParsingError err = ParsingError::None;
        int stack_counter = -1;
        do {
            char c = _kc.wgr.PeekChar(0);
            // If starts like value
            if (HasThisByte(char4{ '{', '[', 't', 'f' }, c) ||
                HasThisByte(char4{ 'n', '"', '-', '-' }, c) ||
                (c >= '0' && c <= '9')) {
                // Already in nested object
                if (stack_counter >= 0) {
                    if (CheckFor::Array && stack[stack_counter].IsValuePossible()) {
                        if (RT::IsLeader()) {
                            stack[stack_counter].SetSeenValue();
                        }
                    } else if (CheckFor::Object && (c == '"' && stack[stack_counter].IsKeyPossible())) {
                        if (RT::IsLeader()) {
                            stack[stack_counter].SetSeenKey();
                        }
                    } else return ParsingError::Other;

                }
            }
            // Only one thread was performing writes. Sync in case of independent thread scheduler
            __syncwarp();
            if (CheckFor::Comma && c == ',') {
                if (stack_counter < 0 || !stack[stack_counter].IsCommaPossible()) return ParsingError::Other;
                if (RT::IsLeader()) {
                    stack[stack_counter].SetSeenComma();
                }
                _kc.wgr.AdvanceBy(1);
            } else if (CheckFor::Braces && c == '{') {
                ++stack_counter;
                if (stack_counter >= STACK_SIZE) return ParsingError::Other;
                if (RT::IsLeader()) {
                    stack[stack_counter].SetObject();
                }
                _kc.wgr.AdvanceBy(1);
            } else if (CheckFor::Braces && c == '}') {
                if (stack_counter < 0 || !stack[stack_counter].CanObjectEnd()) return ParsingError::Other;
                --stack_counter;
                _kc.wgr.AdvanceBy(1);
            } else if (CheckFor::Brackets && c == '[') {
                ++stack_counter;
                if (stack_counter >= STACK_SIZE) return ParsingError::Other;
                if (RT::IsLeader()) {
                    stack[stack_counter].SetArray();
                }
                _kc.wgr.AdvanceBy(1);
            } else if (CheckFor::Brackets && c == ']') {
                if (stack_counter < 0 || !stack[stack_counter].CanArrayEnd()) return ParsingError::Other;
                --stack_counter;
                _kc.wgr.AdvanceBy(1);
            } else if (CheckFor::Quotes && c == '"') {
                err = JsonParse::String(_kc, [](bool _, int __) { return ParsingError::None; });
                if (err != ParsingError::None) return err;
                // If this is a key in an object
                if (stack_counter >= 0 && stack[stack_counter].KeyWasSeen()) {
                    err = Parse::FindNext<':', WS>::KC(_kc).template Do<Parse::StopTag::StopAfter>();
                    if (err != ParsingError::None) return err;
                }
            } else if (CheckFor::Null && c == 'n') {
                //TODO validate if null
                _kc.wgr.AdvanceBy(4);
            } else if (CheckFor::Boolean && (c == 't' || c == 'f')) {
                err = JsonParse::Boolean(_kc, [](auto result) { return; });
                if (err != ParsingError::None) return err;
            } else if (CheckFor::Number && (c == '-' || (c >= '0' && c<= '9'))) {
                err = JsonParsers::Float<float, boost::mp11::mp_true, boost::mp11::mp_true>(_kc, [](auto result) { return ; });
                if (err != ParsingError::None) return err;
            } else err = ParsingError::Other;
            // Only one thread was performing writes. Sync in case of independent thread scheduler
            __syncwarp();
            if (stack_counter > STACK_SIZE) {
                return ParsingError::Other;
            }
            if (stack_counter >= 0) {
                err = Parse::FindNoneWhite<WS>::KC(_kc).template Do<Parse::StopTag::StopAt>();
                if (err != ParsingError::None) return err;
            }
        } while (stack_counter >= 0);
        return ParsingError::None;
    }
}


#endif //META_JSON_PARSER_SKIP_CUH
