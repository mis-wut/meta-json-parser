#ifndef META_JSON_PARSER_DATETIME_CUH
#define META_JSON_PARSER_DATETIME_CUH
#include <type_traits>
#include <cuda_runtime_api.h>
#include <cuda/std/chrono>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/function.hpp>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/cuda_wrapper.cuh>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/meta_utility/is_base_of_template.h>

namespace JsonParsers {
    namespace DatetimeTokens {
        enum class Field {
            Year,
            Month,
            Day,
            Hour,
            Minute,
            Second,
            Millisecond,
            Timezone,
            Text
        };


        template<Field FieldT>
        using FieldConst = std::integral_constant<Field, FieldT>;

        template<Field FieldT>
        using YieldsResult_c = boost::mp11::mp_bool<FieldT != Field::Text>;

        template<class FieldT>
        using YieldsResult = YieldsResult_c<FieldT::value>;

        template<class FieldConstT>
        struct TokenBase {
            using Field = FieldConstT;
        };

        template<class LengthT, class FieldConstT>
        struct ConstLengthToken : public TokenBase<FieldConstT>{
            using Length = LengthT;
        };

        template<class N>
        struct YearDigit : public ConstLengthToken<N, FieldConst<Field::Year>> {};
        template<uint32_t N>
        using YearDigit_c = YearDigit<boost::mp11::mp_int<N>>;

        template<class N>
        struct MonthDigit : public ConstLengthToken<N, FieldConst<Field::Month>> {};
        template<uint32_t N>
        using MonthDigit_c = MonthDigit<boost::mp11::mp_int<N>>;

        template<class N>
        struct DayDigit : public ConstLengthToken<N, FieldConst<Field::Day>> {};
        template<uint32_t N>
        using DayDigit_c = DayDigit<boost::mp11::mp_int<N>>;

        template<class N>
        struct HourDigit : public ConstLengthToken<N, FieldConst<Field::Hour>> {};
        template<uint32_t N>
        using HourDigit_c = HourDigit<boost::mp11::mp_int<N>>;

        template<class N>
        struct MinuteDigit : public ConstLengthToken<N, FieldConst<Field::Minute>> {};
        template<uint32_t N>
        using MinuteDigit_c = MinuteDigit<boost::mp11::mp_int<N>>;

        template<class N>
        struct SecondDigit : public ConstLengthToken<N, FieldConst<Field::Second>> {};
        template<uint32_t N>
        using SecondDigit_c = SecondDigit<boost::mp11::mp_int<N>>;

        template<class N>
        struct MillisecondDigit : public ConstLengthToken<N, FieldConst<Field::Millisecond>> {};
        template<uint32_t N>
        using MillisecondDigit_c = MinuteDigit<boost::mp11::mp_int<N>>;

        template<class MpString>
        struct Text : public ConstLengthToken<boost::mp11::mp_size<MpString>, FieldConst<Field::Text>> {};
    }

    namespace _impl_Datetime {
        template<DatetimeTokens::Field FieldT>
        struct IsField {
            template<class T>
            using fn = boost::mp11::mp_bool<T::Field::value == FieldT>;
        };

        template<class TokensT, DatetimeTokens::Field FieldT>
        using AtMostOne = boost::mp11::mp_bool<
            boost::mp11::mp_count_if_q<TokensT, IsField<FieldT>>::value <= 1
        >;

        template<class LengthT, class KernelContextT>
        __device__ __forceinline__ ParsingError ConstLengthDigitParser(KernelContextT& _kc, int& result) {
            using KC = KernelContextT;
            using RT = typename KC::RT;
            using WorkGroupSize = typename RT::WorkGroupSize;
            constexpr int Length = LengthT::value;
            static_assert(
                RT::WorkGroupSize::value >= LengthT::value,
                "ConstLengthDigitParser supports only work groups with sizes not greater than length."
            );
            char c = _kc.wgr.CurrentChar();
            bool isValid = c >= '0' && c <= '9';
            uint32_t activeThreads = Reduce<uint32_t, WorkGroupSize>(_kc).Reduce((isValid && RT::WorkerId() < Length ? 1 : 0), cub::Sum());
            activeThreads = Scan<uint32_t, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
            if (activeThreads != Length) {
                return ParsingError::Other;
            }
            int value = 0;
            if (RT::WorkerId() < Length) {
                int power = 1;
                for (int i = 0; i < (Length - 1 - static_cast<int>(RT::WorkerId())); ++i) {
                    power *= 10;
                }
                value = static_cast<int>(c - '0') * power;
            }
            _kc.wgr.AdvanceBy(Length);
            result = Reduce<int, WorkGroupSize>(_kc).Reduce(value, cub::Sum());
            return ParsingError::None;
        };
    }

    using DatetimeRequests = boost::mp11::mp_list<
        ReduceRequest<
            int
        >,
        ReduceRequest<
            uint32_t
        >,
        ScanRequest<
            uint32_t
        >
    >;

    /**
     * @tparam MillisecondT true_type or false_type, For seconds output will be a uint32_t,
     * for milliseconds it will be uint64_t
     * @tparam TokensT
     * @tparam KernelContextT
     * @tparam CallbackFnT
     * @param _kc
     * @param fn
     * @return
     */
    template<class MillisecondT, class TokensT, class OutTypeT, class KernelContextT, class CallbackFnT>
    __device__ INLINE_METHOD ParsingError Datetime(KernelContextT& _kc, CallbackFnT&& fn) {
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Year>::value, "Year can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Month>::value, "Month can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Day>::value, "Day can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Hour>::value, "Hour can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Minute>::value, "Minute can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Second>::value, "Second can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Millisecond>::value, "Millisecond can occur at most one time.");
        static_assert(_impl_Datetime::AtMostOne<TokensT, DatetimeTokens::Field::Timezone>::value, "Timezone can occur at most one time.");
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using Out = OutTypeT;
        using DurationMultiplier = boost::mp11::mp_if<
            MillisecondT,
            std::integral_constant<Out, 1000>,
            std::integral_constant<Out, 1>
        >;
        using Duration = boost::mp11::mp_if<
            MillisecondT,
            cuda::std::chrono::milliseconds,
            cuda::std::chrono::seconds
        >;
        cuda::std::chrono::year year(1970);
        cuda::std::chrono::month month(1);
        cuda::std::chrono::day day(1);
        //sizeof( cuda::std::chrono::hh_mm_ss<Duration> ) == 40
        Out result = 0;

        ParsingError err = ParsingError::None;
        boost::mp11::mp_for_each<TokensT>([&](auto token){
            if (err != ParsingError::None)
                return;
            using Token = decltype(token);
            using IsConstLengthToken = is_base_of_template<DatetimeTokens::ConstLengthToken, Token>;
            int value = 0;
            static_assert(IsConstLengthToken::value, "Unsupported datetime token.");
            if constexpr (IsConstLengthToken::value) {
                using DoesYieldValue = DatetimeTokens::YieldsResult<typename Token::Field>;
                using IsTextToken = boost::mp11::mp_similar<DatetimeTokens::Text<boost::mp11::mp_list<>>, Token>;
                static_assert(DoesYieldValue::value || IsTextToken::value, "Unsupported const length datetime token.");
                if constexpr (DoesYieldValue::value) {
                    if constexpr (MillisecondT::value || Token::Field::value != DatetimeTokens::Field::Millisecond) {
                        err = _impl_Datetime::ConstLengthDigitParser<Token::Length>(_kc, value);
                    } else {
                        //Do not parse milliseconds if output is seconds timestamp
                        //Allows for malformed input since milliseconds will not be validated
                        _kc.wgr.AdvanceBy(Token::Length::value);
                    }
                    if constexpr (Token::Field::value == DatetimeTokens::Field::Year) {
                        year = static_cast<cuda::std::chrono::year>(value);
                    } else if constexpr (Token::Field::value == DatetimeTokens::Field::Month) {
                        month = static_cast<cuda::std::chrono::month>(value);
                    } else if constexpr (Token::Field::value == DatetimeTokens::Field::Day) {
                        day = static_cast<cuda::std::chrono::day>(value);
                    } else if constexpr (Token::Field::value == DatetimeTokens::Field::Hour) {
                        result += value * (60 * 60 * DurationMultiplier::value);
                    } else if constexpr (Token::Field::value == DatetimeTokens::Field::Minute) {
                        result += value * (60 * DurationMultiplier::value);
                    } else if constexpr (Token::Field::value == DatetimeTokens::Field::Second) {
                        result += value * DurationMultiplier::value;
                    } else if constexpr (Token::Field::value == DatetimeTokens::Field::Millisecond) {
                        result += value;
                    }
                } else if constexpr (IsTextToken::value) {
                    //Right now characters aren't being validated.
                    _kc.wgr.AdvanceBy(Token::Length::value);
                }
            }
        });
        if (err != ParsingError::None)
            return err;
        result += cuda::std::chrono::duration_cast<Duration>(
                cuda::std::chrono::sys_days {
                    cuda::std::chrono::year_month_day { year, month, day}
                }.time_since_epoch()
            ).count();
        if (RT::WorkerId() == 0)
            fn(result);
        return ParsingError::None;
    }
}

#endif //META_JSON_PARSER_DATETIME_CUH
