#ifndef META_JSON_PARSER_JSTRING_CUSTOM_CUH
#define META_JSON_PARSER_JSTRING_CUSTOM_CUH
#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/output_printer.cuh>

template<class TagT>
struct ExampleStringTransformFunctor
{
    using OutputType = int;
    using type = ExampleStringTransformFunctor;
    using Tag = TagT;
    //using Printer = MyCustomPrinter;
    using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutputType>>;
    using MemoryRequests = boost::mp11::mp_list<>;

    //Place your state variables inside functor class

    inline __device__ ExampleStringTransformFunctor() {};

    template<class KernelContextT>
    inline __device__ ParsingError operator()(KernelContextT& kc, bool& escaped, int& activeChars) {
        //Here goes computation
        return ParsingError::None;
    };

    template<class KernelContextT>
    inline __device__ ParsingError finalize(KernelContextT& kc) {
        //Here goes final computation

        //Save result using OutputManager
        using RT = typename KernelContextT::RT;
        if (RT::WorkerId() == 0)
            kc.om.template Get<KernelContextT, TagT>() = 0;

        return ParsingError::None;
    };
};

/**
 * @tparam TransformFunctor see ExampleStringTransformFunctor for reference how to implement functor.
 *
 * @see ExampleStringTransformFunctor
 */
template<class StringTransformFunctorT>
struct JStringCustom
{
    using type = JStringCustom<StringTransformFunctorT>;
    using Tag = typename StringTransformFunctorT::Tag;
    using MemoryRequests = boost::mp11::mp_append<
            JsonParse::StringRequests,
            typename StringTransformFunctorT::MemoryRequests
    >;
    using OutputRequests = typename StringTransformFunctorT::OutputRequests;
    using Printer = GetPrinter<StringTransformFunctorT>;

#ifdef HAVE_LIBCUDF
    using CudfColumnConverter = CudfCategoricalColumn<type, typename StringTransformFunctorT::OutputType>;
#endif

    template<class KernelContextT>
    static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
    {
        using KC = KernelContextT;
        StringTransformFunctorT fc;
        ParsingError err = JsonParse::String(kc, [&fc, &kc](bool& escaped, int& activeChars){ return fc(kc, escaped, activeChars); });
        if (err != ParsingError::None)
            return err;
        return fc.finalize(kc);
    }
};
#endif //META_JSON_PARSER_JSTRING_CUSTOM_CUH
