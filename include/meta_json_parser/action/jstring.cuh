#pragma once
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/static_buffer.h>
#include <meta_json_parser/output_manager.cuh>

//Only parsing/validation
struct JString
{
	using OutputRequests = boost::mp11::mp_list<>;
	using MemoryRequests = JsonParse::StringRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		return JsonParse::String<KC>(kc)([](bool&, int&){ return ParsingError::None; });
	}
};

template<class BytesT, class TagT>
struct JStringStaticCopy
{
	static_assert(BytesT::value > 0, "BytesT must be at greater than 0");

	using OutputRequests = boost::mp11::mp_list<OutputRequest<TagT, StaticBuffer_c<BytesT::value>>>;
	using MemoryRequests = JsonParse::StringRequests;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
        using KC = KernelContextT;
		using RT = typename KC::RT;
		char (&result)[BytesT::value] = kc.om.template Get<KernelContextT, TagT>().template Alias<char[BytesT::value]>();
		uint32_t offset = 0;
		ParsingError err = JsonParse::String<KC>(kc)([&](bool& isEscaped, int& activeThreads) {
			uint32_t worker_offset = offset + RT::WorkerId();
			char c = RT::WorkerId() < activeThreads ? kc.wgr.CurrentChar() : '\0';
			if (worker_offset < BytesT::value)
				result[worker_offset] = c;
			offset += RT::GroupSize();
			return ParsingError::None;
		});
		if (err != ParsingError::None)
			return err;
		while (offset < BytesT::value)
		{
			uint32_t worker_offset = offset + RT::WorkerId();
			if (worker_offset < BytesT::value)
				result[worker_offset] = '\0';
			offset += RT::GroupSize();
		}
		return ParsingError::None;
	}
};
