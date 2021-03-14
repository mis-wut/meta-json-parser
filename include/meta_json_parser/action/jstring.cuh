#pragma once
#include <meta_json_parser/json_parse.cuh>

//Only parsing/validation
struct JString
{
	using OutputRequests = boost::mp11::mp_list<>;
	using MemoryRequests = JsonParse::StringRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		using RT = typename KernelContextT::RT;
		return JsonParse::String<typename RT::WorkGroupSize>::KC(kc)();
	}
};