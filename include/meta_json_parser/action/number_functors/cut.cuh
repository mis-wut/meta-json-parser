#ifndef META_JSON_PARSER_CUT_CUH
#define META_JSON_PARSER_CUT_CUH
#include <type_traits>
#include <ratio>
#include <boost/mp11/list.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/integer_sequence.hpp>
#include <cuda_runtime_api.h>
#include <meta_json_parser/meta_utility/safe_drop.h>
#include <meta_json_parser/meta_utility/option_getter.cuh>

struct CutFunctorOptions {
    struct Labels {
        struct BinBased {};
        using Default = BinBased;
    };

    struct OutOfRange {
        struct NanValue {
            template<class CutFunctorT, class NumberT>
            __forceinline__ __host__ __device__ constexpr static typename CutFunctorT::OutType Handle(NumberT val) {
                static_assert(
                    std::is_floating_point_v<typename CutFunctorT::OutType>,
                    "OutOfRange::NanValue cab be used only with floating point labels"
                );
                return static_cast<typename CutFunctorT::OutType>(NAN);
            }
        };
        /**
         * Constant value.
         * @tparam ValueT Must have constexpr ::value member. Like std::integral_constant
         */
        template<class ValueT>
        struct ConstValue {
            template<class CutFunctorT, class NumberT>
            __forceinline__ __host__ __device__ constexpr static typename CutFunctorT::OutType Handle(NumberT val) {
                return static_cast<typename CutFunctorT::OutType>(ValueT::value);
            }
        };
        using Default = NanValue;
    };

    struct Right {
        using True = std::true_type;
        using False = std::false_type;
        using Default = True;
    };

    OPTION_GETTER(CutFunctorOptions, Labels)
    OPTION_GETTER(CutFunctorOptions, OutOfRange)
    OPTION_GETTER(CutFunctorOptions, Right)
};

template<class T>
struct CutBin {
    constexpr static __device__ __forceinline__ auto GetValue() { return T::value; }
};

template<intmax_t Num, intmax_t Den>
struct CutBin<std::ratio<Num, Den>> {
    constexpr static __device__ __forceinline__ double GetValue() { return double(Num) / double(Den); }
};

template<uint8_t value>
using CutBinUint8_c = CutBin<std::integral_constant<uint8_t, value>>;

template<int8_t value>
using CutBinInt8_c = CutBin<std::integral_constant<int8_t, value>>;

template<uint16_t value>
using CutBinUint16_c = CutBin<std::integral_constant<uint16_t, value>>;

template<int16_t value>
using CutBinInt16_c = CutBin<std::integral_constant<int16_t, value>>;

template<uint32_t value>
using CutBinUint32_c = CutBin<std::integral_constant<uint32_t, value>>;

template<int32_t value>
using CutBinInt32_c = CutBin<std::integral_constant<int32_t, value>>;

template<uint64_t value>
using CutBinUint64_c = CutBin<std::integral_constant<uint64_t, value>>;

template<int64_t value>
using CutBinInt64_c = CutBin<std::integral_constant<int64_t, value>>;

template<intmax_t Num, intmax_t Den>
using CutBinReal_c = CutBin<std::ratio<Num, Den>>;

namespace meta_json_parser::details {
    template<class BinA, class BinB>
    struct LessBin {
        static_assert(boost::mp11::mp_similar<BinA, CutBin<boost::mp11::mp_int<0>>>::value, "BinA must be a CutBin.");
        static_assert(boost::mp11::mp_similar<BinB, CutBin<boost::mp11::mp_int<0>>>::value, "BinB must be a CutBin.");
    };
    template<class ContentA, class ContentB>
    struct LessBin<CutBin<ContentA>, CutBin<ContentB>> {
        using BinA = CutBin<ContentA>;
        using BinB = CutBin<ContentB>;
        constexpr static bool isLess = BinA::GetValue() < BinB::GetValue();
        using type = boost::mp11::mp_bool<isLess>;
    };
}

template<class T>
using CutLabel = CutBin<T>;

template<class BinsT, class OptionsT = boost::mp11::mp_list<>>
struct CutFunctor {
    using type = CutFunctor<BinsT>;
    using Bins = BinsT;
    using Options = OptionsT;
    using LabelsOpt = CutFunctorOptions::GetLabels<OptionsT>;
    using OutOfRangeOpt = CutFunctorOptions::GetOutOfRange<OptionsT>;
    using RightOpt = CutFunctorOptions::GetRight<OptionsT>;
    using BinsNumber = boost::mp11::mp_size<BinsT>;
    static_assert(BinsNumber::value >= 2, "There must be at least two bins.");
    using AscendingBins = boost::mp11::mp_rename<
        boost::mp11::mp_pairwise_fold_q<
            Bins,
            boost::mp11::mp_quote_trait<meta_json_parser::details::LessBin>
        >,
        boost::mp11::mp_all
    >;
    static_assert(AscendingBins::value, "Bins must be ascending");
    using Labels = boost::mp11::mp_eval_if_not<
        std::is_same<LabelsOpt, CutFunctorOptions::Labels::BinBased>,
        LabelsOpt,
        boost::mp11::mp_drop,
        Bins,
        boost::mp11::mp_int<1>
    >;
private:
    template<class LabelT>
    using _impl_GetLabelType = decltype(LabelT::GetValue());
public:
    using LabelTypes = boost::mp11::mp_transform<
        _impl_GetLabelType,
        Labels
    >;
    using SameType = boost::mp11::mp_bool<(1 == boost::mp11::mp_size<boost::mp11::mp_unique<LabelTypes>>::value)>;
    static_assert(SameType::value, "Labels must have the same return type from GetValue().");
    using OutType = boost::mp11::mp_first<LabelTypes>;

    template<class NumberT>
    __forceinline__ __device__ OutType operator()(NumberT val) const {
        using LowestBin = boost::mp11::mp_first<Bins>;
        using HighestBin = boost::mp11::mp_back<Bins>;
        if ((RightOpt::value && (val <= LowestBin::GetValue() || val > HighestBin::GetValue())) ||
            (!RightOpt::value && (val < LowestBin::GetValue() || val >= HighestBin::GetValue()))) {
            return OutOfRangeOpt::template Handle<type>(val);
        }
        bool found = false;
        OutType result{0};
        boost::mp11::mp_for_each<boost::mp11::mp_iota_c<BinsNumber::value - 1>>([&](auto idx){
            using Idx = decltype(idx);
            using Bin = boost::mp11::mp_at_c<Bins, Idx::value + 1>;
            using Label = boost::mp11::mp_at_c<Labels, Idx::value>;
            if (found)
                return;
            if ((RightOpt::value && val <= Bin::GetValue()) ||
                (!RightOpt::value && val < Bin::GetValue())) {
                result = Label::GetValue();
                found = true;
            }
        });
        return result;
    }
};

#endif //META_JSON_PARSER_CUT_CUH
