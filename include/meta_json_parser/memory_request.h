#pragma once
#include <boost/mp11/integral.hpp>
#include <meta_json_parser/static_buffer.h>

namespace MemoryType
{
	struct Shared {};
}

namespace MemoryUsage
{
	struct ReadOnly {};
	struct AtomicUsage {};
	struct ActionUsage {};
}
//TODO add alignment
//TODO add memory type (global, shared, const)
//TODO add usage (read-only, atomic-usage-buffer, action-usage-buffer)
template<class SizeT, class MemoryUsageT, class MemoryTypeT = MemoryType::Shared>
struct MemoryRequest
{
	using Size = SizeT;
	using Buffer = StaticBuffer<Size>;
	using MemoryType = MemoryTypeT;
	using MemoryUsage = MemoryUsageT;
};

template<int SizeT, class MemoryUsageT, class MemoryTypeT = MemoryType::Shared>
using MemoryRequest_c = MemoryRequest<boost::mp11::mp_int<SizeT>, MemoryUsageT, MemoryTypeT>;
