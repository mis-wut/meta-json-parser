#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>

struct VoidAction
{
	using OutputRequests = boost::mp11::mp_list<>;
	using MemoryRequests = boost::mp11::mp_list<>;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		return ParsingError::None;
	}
};
