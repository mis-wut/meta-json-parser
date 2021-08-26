#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>

template<class OutT, class TagT>
struct JNumber
{
	using Tag = TagT;
	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, OutT>>;
	using MemoryRequests = JsonParse::UnsignedIntegerRequests<OutT>;
	static_assert(std::is_integral_v<OutT>, "OutT must be integral.");
	static_assert(std::is_unsigned_v<OutT>, "OutT must be unsigned.");

#if defined(HAVE_LIBCUDF)
	using CudfConverter = CudfNumericColumn<OutT>;
#endif

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		return JsonParse::UnsignedInteger<OutT>(kc, [&](auto&& result) {
			kc.om.template Get<KernelContextT, TagT>() = result;
		});
	}
};
