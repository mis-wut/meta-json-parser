#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>

template<class OutT, class TagT>
struct JBool
{
	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutT>>;
	using MemoryRequests = JsonParse::BooleanRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		using RT = typename KernelContextT::RT;
		return JsonParse::Boolean<typename RT::WorkGroupSize>(kc, kc.om.template Get<KernelContextT, TagT>());
	}
};
