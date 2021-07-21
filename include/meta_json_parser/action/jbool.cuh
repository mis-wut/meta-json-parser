#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>

#if defined(HAVE_LIBCUDF)
#include <boost/mp11/utility.hpp>
#endif

template<class OutT, class TagT>
struct JBool
{
	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutT>>;
	using MemoryRequests = JsonParse::BooleanRequests;

#if defined(HAVE_LIBCUDF)
	// check during compile time if the TagT is equivalent to 'bool' type
	// and can be reinterpret_cast-ed.
	using CudfConverter = boost::mp11::mp_if_c<
		sizeof(TagT) == sizeof(bool),
		CudfBoolColumn,
		std::false_type
	>;
#endif

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		using RT = typename KernelContextT::RT;
		return JsonParse::Boolean<typename RT::WorkGroupSize>(kc, kc.om.template Get<KernelContextT, TagT>());
	}
};
