#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/output_printer.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>

template<class OutT, class TagT>
struct JBool
{
	using type = JBool<OutT, TagT>;
	using Tag = TagT;
	using Printer = BoolPrinter<type>;
	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutT>>;
	using MemoryRequests = JsonParse::BooleanRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		return JsonParse::Boolean(kc, [&](auto&& result) {
			kc.om.template Get<KernelContextT, TagT>() = result;
		});
	}
};
