#ifndef META_JSON_PARSER_POLYNOMIAL_ROLLING_HASH_CUH
#define META_JSON_PARSER_POLYNOMIAL_ROLLING_HASH_CUH

template<class MultiplierT, class ModulusT, class OutputT, class TagT>
class PolynomialRollingHashFunctor
{
public:
    using type = PolynomialRollingHashFunctor<MultiplierT, ModulusT, OutputT, TagT>;
    using OutputType = OutputT;
    using HashType = size_t;
    using Tag = TagT;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutputType>>;
    using MemoryRequests = boost::mp11::mp_list<ReduceRequest<HashType>>;

    HashType hash;
    HashType power;

    inline __device__ PolynomialRollingHashFunctor() :
        hash(static_cast<HashType>(0)),
        power(static_cast<HashType>(1)) {};

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

        HashType final_hash = Reduce<HashType, RT::WorkGroupSize>(kc).Reduce(
            (hash * (RT::WorkerId() + 1)) % ModulusT::value,
            SumModulo<ModulusT>()
        );

        if (RT::WorkerId() == 0)
            kc.om.template Get<KernelContextT, TagT>() = static_cast<OutputType>(final_hash);
        return ParsingError::None;
    };
};

#endif //META_JSON_PARSER_POLYNOMIAL_ROLLING_HASH_CUH
