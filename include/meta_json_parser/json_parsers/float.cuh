#ifndef HASH_PY_FLOAT_CUH
#define HASH_PY_FLOAT_CUH
#include <cuda_runtime_api.h>
#include <type_traits>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/cuda_wrapper.cuh>
#include <meta_json_parser/meta_utility/length_representation.h>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/parsing_error.h>

namespace JsonParsers {
    template<class OutTypeT>
    using FloatOperationType = boost::mp11::mp_if<
        std::is_same<OutTypeT, double>,
        double,
        float
    >;

    template<class OutTypeT>
    using FloatRequests = boost::mp11::mp_list<
        ReduceRequest<
            int
        >,
        ReduceRequest<
            FloatOperationType<OutTypeT>
        >,
        ReduceRequest<
            float
        >,
        ReduceRequest<
            uint32_t
        >,
        ScanRequest<
            int
        >
    >;

    /**
     * Parses sequence of digit characters. Stops at first non digit character.
     * Correct value is returned by worker with id = 0.
     * @tparam ValueType
     * @tparam KernelContextT
     * @param _kc
     * @param seed
     * @return
     */
    template<class ValueType, class KernelContextT>
    __device__ INLINE_METHOD ValueType DigitLoop(KernelContextT& _kc, ValueType seed) {
        static_assert(std::is_floating_point_v<ValueType>, "DigitLoop is designed only for floating point types.");
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;

        ValueType output = seed;
        int activeThreads = WorkGroupSize::value;

        while (activeThreads == WorkGroupSize::value)
        {
            char c = _kc.wgr.CurrentChar();
            bool isEnd = c < '0' || c > '9';
            activeThreads = Reduce<int, WorkGroupSize>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSize::value), cub::Min());
            activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
            // No digits/charactes in each next workgroup pass
            if (activeThreads == 0)
                break;

            // TODO Temporary solution. Should be replaced with lookup table
            output *= template_cuda_math::exp10(static_cast<ValueType>(activeThreads));

            if (RT::WorkerId() < activeThreads)
            {
                output += static_cast<ValueType>(c - '0') *
                        template_cuda_math::exp10(static_cast<ValueType>(activeThreads - RT::WorkerId() - 1));
            }
            _kc.wgr.AdvanceBy(activeThreads);
        }

        return Reduce<ValueType, WorkGroupSize>(_kc).Reduce(output, cub::Sum());
    }

    template<class ValueType, class KernelContextT>
    __device__ INLINE_METHOD ValueType FractionLoop(KernelContextT& _kc, uint32_t& readCharacters) {
        static_assert(std::is_integral_v<ValueType>, "FractionLoop is designed only for integral types.");
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;

        // divide by 10 because: len(str(2**64)) -> 20, but log2(int('9' * 20)) ~= 66.43 > 64.
        //                       len(str(2**32)) -> 10, but log2(int('9' * 10)) ~= 33.22 > 32.
        //                       len(str(2**16)) ->  5, but log2(int('9' *  5)) ~= 16.61 > 16.
        //                       len(str(2**8 )) ->  3, but log2(int('9' *  3)) ~=  9.96 >  8.
        // So to make sure that every possible value will fit we divide value by 10 to reduce repr size by 1,
        // but still have repr > 0.
        using MaxReprT = LengthRepresentation_c<std::numeric_limits<ValueType>::max() / 10>;

        ValueType output = 0;
        int activeThreads = WorkGroupSize::value;
        readCharacters = 0;

        while (activeThreads == WorkGroupSize::value) {
            char c = _kc.wgr.CurrentChar();
            bool isEnd = c < '0' || c > '9';

            activeThreads = Reduce<int, WorkGroupSize>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSize::value), cub::Min());
            activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);

            if (MaxReprT::value < activeThreads + readCharacters) {
                activeThreads = MaxReprT::value - readCharacters;
            }

            // No digits/charactes
            if (activeThreads == 0)
                break;

            if (output != 0)
            {
                // TODO Temporary solution. Should be replaced with lookup table
                ValueType power = 1;
                for (int i = 0; i < activeThreads; ++i)
                    power *= ValueType(10);
                output *= power;
            }

            if (RT::WorkerId() < activeThreads)
            {
                // TODO Temporary solution. Should be replaced with lookup table
                ValueType power = 1;
                for (int i = 0; i < (activeThreads - RT::WorkerId() - 1); ++i)
                    power *= ValueType(10);
                output += static_cast<ValueType>(c - '0') * power;
            }
            readCharacters += activeThreads;
            _kc.wgr.AdvanceBy(activeThreads);
        }

        activeThreads = WorkGroupSize::value;
        while (activeThreads == WorkGroupSize::value) {
            char c = _kc.wgr.CurrentChar();
            bool isEnd = c < '0' || c > '9';
            activeThreads = Reduce<int, WorkGroupSize>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSize::value), cub::Min());
            activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
            _kc.wgr.AdvanceBy(activeThreads);
        }

        return Reduce<ValueType, WorkGroupSize>(_kc).Reduce(output, cub::Sum());
    }

    template<class OutTypeT, class SignT, class ExponentT, class KernelContextT, class CallbackFnT>
    __device__ INLINE_METHOD ParsingError Float(KernelContextT& _kc, CallbackFnT&& fn) {
        static_assert(std::is_arithmetic_v<OutTypeT>, "OutTypeT must be arithmetic.");
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;
        using OP = FloatOperationType<OutTypeT>;
        static_assert(WorkGroupSize::value >= 4, "WorkGroup must have a size of at least 4.");
        using Sign = SignT;
        using Exponent = ExponentT;

        bool minus = false;

        // TODO var 'minus' should exist only if Sign::value == true.
        //      Otherwise function shouldn't use an additional register for it.
        if constexpr (Sign::value) {
            if (_kc.wgr.PeekChar(0) == '-') {
                minus = true;
                _kc.wgr.AdvanceBy(1);
            }
        }

        OP digits = OP(0);

        if (_kc.wgr.PeekChar(0) < '0' || _kc.wgr.PeekChar(0) > '9')
            return ParsingError::Other;

        //Finish loop
        if (_kc.wgr.PeekChar(0) == '0') {
            _kc.wgr.AdvanceBy(1);
        } else {
            digits = DigitLoop<OP>(_kc, 0);
        }

        digits = template_cuda_math::nearbyint(digits);

        uint32_t fraction = 0;
        uint32_t fractionLen = 0;

        if (_kc.wgr.PeekChar(0) == '.') {
            _kc.wgr.AdvanceBy(1);
            if (_kc.wgr.PeekChar(0) < '0' || _kc.wgr.PeekChar(0) > '9')
                return ParsingError::Other;
            fraction = FractionLoop<uint32_t>(_kc, fractionLen);
        }

        // TODO var 'exp' should exist only if Exponent::value == true.
        //      Otherwise function shouldn't use an additional register for it.
        float exp = 0;

        if constexpr (Exponent::value) {
            if (_kc.wgr.PeekChar(0) == 'e' || _kc.wgr.PeekChar(0) == 'E') {
                _kc.wgr.AdvanceBy(1);
                bool minus_exp = false;
                switch (_kc.wgr.PeekChar(0)) {
                    case '-':
                        minus_exp = true;
                    case '+':
                        _kc.wgr.AdvanceBy(1);
                }
                if (_kc.wgr.PeekChar(0) < '0' || _kc.wgr.PeekChar(0) > '9')
                    return ParsingError::Other;
                exp = DigitLoop<float>(_kc, 0);
                if (minus_exp)
                    exp = -exp;
            }
        }

        if (KC::RT::WorkerId() == 0) {
            // corner case
            // 0.123456E39, 0.123456 * exp10(39) = 0.123456 * (float)inf = inf
            if (digits == 0) {
                exp -= 1;
                fractionLen -= 1;
            }

            digits += fraction / template_cuda_math::exp10(static_cast<OP>(fractionLen));

            if (exp < 0) {
                // At this point digits are still without sign
                OP digitLog10 = template_cuda_math::floor(template_cuda_math::log10(digits));
                if (digitLog10 > 0) {
                    // Assumption: exp is an integer at this point
                    exp += digitLog10;
                    digits /= template_cuda_math::exp10(digitLog10);
                }
            }

            OP final_exp = template_cuda_math::exp10(static_cast<OP>(exp));

            if (minus)
                digits = -digits;
            if (digits == 0)
                fn(digits);
            else
                fn(digits * final_exp);
        }

        return ParsingError::None;
    }

#endif //HASH_PY_FLOAT_CUH
}

