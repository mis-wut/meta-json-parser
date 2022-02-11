#ifndef META_JSON_PARSER_POLYNOMIAL_ROLLING_HASH_MATCHER_CUH
#define META_JSON_PARSER_POLYNOMIAL_ROLLING_HASH_MATCHER_CUH
#include <type_traits>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/map.hpp>
#include <meta_json_parser/meta_utility/safe_drop.h>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/parsing_error.h>

namespace meta_json_parser {
    namespace details {
        template<uint64_t N>
        using HashUint = std::integral_constant<uint64_t, N>;

        template<class AccumulatorT, class PowerT, class MultiplierT, class ModulusT, class WorkGroupSize, class ThreadId, class StringT>
        struct PolynomialRollingHash_impl_thread {
            using IsActive = boost::mp11::mp_bool<ThreadId::value < boost::mp11::mp_size<StringT>::value>;
            using Char = boost::mp11::mp_eval_if_not<
                IsActive,
                boost::mp11::mp_int<' '>,
                boost::mp11::mp_at,
                StringT,
                ThreadId
            >;
            using Rest = safe_drop<StringT, WorkGroupSize>;
            using NextAccumulator = HashUint<
                (AccumulatorT::value + (Char::value - ' ' + 1) * PowerT::value) % ModulusT::value
            >;
            using NextPower = HashUint<
                (PowerT::value * MultiplierT::value) % ModulusT::value
            >;

            using Inner = PolynomialRollingHash_impl_thread<
                boost::mp11::mp_if<IsActive, NextAccumulator, AccumulatorT>,
                boost::mp11::mp_if<IsActive, NextPower, PowerT>,
                MultiplierT,
                ModulusT,
                WorkGroupSize,
                ThreadId,
                Rest
            >;
            using type = typename Inner::type;
        };

        template<class AccumulatorT, class PowerT, class MultiplierT, class ModulusT, class WorkGroupSize, class ThreadId>
        struct PolynomialRollingHash_impl_thread<AccumulatorT, PowerT, MultiplierT, ModulusT, WorkGroupSize, ThreadId, boost::mp11::mp_list<>> {
            using NextAccumulator = HashUint<
                (AccumulatorT::value * (ThreadId::value + 1)) % ModulusT::value
            >;
            using type = NextAccumulator;
        };

        template<class ModulusT>
        struct ModuloFold {
            template<class Aggregate, class Value>
            using fn = HashUint<
                (Aggregate::value + Value::value) % ModulusT::value
            >;
        };

        template<class SeedT, class MultiplierT, class ModulusT, class WorkGroupSize, class StringT>
        struct PolynomialRollingHash_impl {
            using ThreadsIds = boost::mp11::mp_iota<WorkGroupSize>;
            using ThreadHashes = boost::mp11::mp_transform_q<
                boost::mp11::mp_bind_q<
                    boost::mp11::mp_quote_trait<PolynomialRollingHash_impl_thread>,
                    SeedT,
                    HashUint<1>,
                    MultiplierT,
                    ModulusT,
                    WorkGroupSize,
                    boost::mp11::_1,
                    StringT
                >,
                ThreadsIds
            >;

            using type = boost::mp11::mp_fold_q<
                ThreadHashes,
                HashUint<0>,
                ModuloFold<ModulusT>
            >;
        };
    }
}

template<class MultiplierT, class ModulusT, class WorkGroupSize, class StringT>
using PolynomialRollingHash = typename meta_json_parser::details::PolynomialRollingHash_impl<
    meta_json_parser::details::HashUint<0>,
    MultiplierT,
    ModulusT,
    WorkGroupSize,
    StringT
>::type;

template<class MultiplierT, class ModulusT, class WorkGroupSize>
struct PolynomialRollingHashMetafunctor {
    template<class StringT>
    using fn = PolynomialRollingHash<MultiplierT, ModulusT, WorkGroupSize, StringT>;
};

template<class ModulusT>
struct SumModulo
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return (a + b) % ModulusT::value;
    }
};

template<class MultiplierT, class ModulusT, class HashMappingT, class TagT>
class PolynomialRollingHashMatcher
{
    template<class T>
    using GetValueType = typename T::value_type;

    using Keys = boost::mp11::mp_transform<
            boost::mp11::mp_first,
            HashMappingT
    >;
    using Categories = boost::mp11::mp_transform<
            boost::mp11::mp_second,
            HashMappingT
    >;
    using CategoriesValueTypes = boost::mp11::mp_transform<
            GetValueType,
            Categories
    >;
public:
    using type = PolynomialRollingHashMatcher<MultiplierT, ModulusT, HashMappingT, TagT>;
    using OutputType = boost::mp11::mp_first<CategoriesValueTypes>;
    static_assert(boost::mp11::mp_size<HashMappingT>::value > 0, "HashMappingT need to have at least one entry.");
    static_assert(boost::mp11::mp_is_map<HashMappingT>::value, "HashMappingT needs to be a map.");
    static_assert(
        boost::mp11::mp_size<boost::mp11::mp_unique<CategoriesValueTypes>>::value == 1,
        "Categories need to have the same value_type."
    );
    static_assert(
        !boost::mp11::mp_contains<Categories, std::integral_constant<OutputType, static_cast<OutputType>(0)>>::value,
        "Categories cannot have value 0. It is reserved for failed match."
    );
    using HashType = size_t;
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutputType>>;
    using MemoryRequests = boost::mp11::mp_list<ReduceRequest<HashType>>;

    HashType hash;
    HashType power;

    inline __device__ PolynomialRollingHashMatcher() : hash(static_cast<HashType>(0)), power(static_cast<HashType>(1)) {};

    template<class KernelContextT>
    inline __device__ ParsingError operator()(KernelContextT& kc, bool& escaped, int& activeChars) {
        using RT = typename KernelContextT::RT;
        if (!escaped && RT::WorkerId() < activeChars) {
            hash = (hash + (static_cast<HashType>(kc.wgr.CurrentChar() - ' ') + 1) * power) % ModulusT::value;
        }
        power = (power * MultiplierT::value) % ModulusT::value;
        return ParsingError::None;
    };

    template<class KernelContextT>
    inline __device__ ParsingError finalize(KernelContextT& kc) {
        using RT = typename KernelContextT::RT;
        using Hashes = boost::mp11::mp_transform_q<
            PolynomialRollingHashMetafunctor<
                MultiplierT,
                ModulusT,
                typename RT::WorkGroupSize
            >,
            Keys
        >;
        static_assert(
            boost::mp11::mp_size<boost::mp11::mp_unique<Hashes>>::value == boost::mp11::mp_size<Keys>::value,
            "There is a hash conflict for one of the workgroup size."
        );

        HashType final_hash = Reduce<HashType, RT::WorkGroupSize>(kc).Reduce(
            (hash * (RT::WorkerId() + 1)) % ModulusT::value,
            SumModulo<ModulusT>()
        );

        OutputType result = static_cast<OutputType>(0);
        boost::mp11::mp_for_each<HashMappingT>([&](auto mapping) {
            using Key = boost::mp11::mp_first<decltype(mapping)>;
            using Hash = PolynomialRollingHash<MultiplierT, ModulusT, typename RT::WorkGroupSize, Key>;
            using CategoryValue = boost::mp11::mp_second<decltype(mapping)>;
            if (final_hash == static_cast<HashType>(Hash::value))
                result = CategoryValue::value;
        });
        if (RT::WorkerId() == 0)
            kc.om.template Get<KernelContextT, TagT>() = result;
        return ParsingError::None;
    };
};

#endif //META_JSON_PARSER_POLYNOMIAL_ROLLING_HASH_MATCHER_CUH
