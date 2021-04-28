#pragma once
#include <boost/mp11/integral.hpp>
#include <boost/mp11/bind.hpp>
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

template<class RequestT>
using GetRequestSizeV = typename RequestT::Size;

template<class RequestT, class RT>
using GetRequestSizeF = typename RequestT::SizeF::fn<RT>;

template<class RequestT, class RT>
using GetRequestSize = typename boost::mp11::mp_cond<
	boost::mp11::mp_valid<GetRequestSizeV, RequestT>, boost::mp11::mp_defer<GetRequestSizeV, RequestT>,
	boost::mp11::mp_valid<GetRequestSizeF, RequestT, RT>, boost::mp11::mp_defer<GetRequestSizeF, RequestT, RT>
>::type;

template<class RequestT>
using GetRequestBufferV = typename RequestT::Buffer;

template<class RequestT, class RT>
using GetRequestBufferF = typename RequestT::BufferF::fn<RT>;

template<class RequestT, class RT>
using GetRequestBuffer = typename boost::mp11::mp_cond<
	boost::mp11::mp_valid<GetRequestBufferV, RequestT>, boost::mp11::mp_defer<GetRequestBufferV, RequestT>,
	boost::mp11::mp_valid<GetRequestBufferF, RequestT, RT>, boost::mp11::mp_defer<GetRequestBufferF, RequestT, RT>
>::type;

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

template<class SizeQ, class MemoryUsageT, class MemoryTypeT = MemoryType::Shared>
struct MemoryRequestRT_q
{
	using SizeF = SizeQ;

	using BufferF = boost::mp11::mp_bind<
		StaticBuffer,
		boost::mp11::mp_bind_q<
			SizeF,
			boost::mp11::_1
		>
	>;

	using MemoryType = MemoryTypeT;
	using MemoryUsage = MemoryUsageT;
};

template<template <class ...> class SizeF, class MemoryUsageT, class MemoryTypeT = MemoryType::Shared>
using MemoryRequestRT = MemoryRequestRT_q<boost::mp11::mp_quote<SizeF>, MemoryUsageT, MemoryTypeT>;

template<int SizeT, class MemoryUsageT, class MemoryTypeT = MemoryType::Shared>
using MemoryRequest_c = MemoryRequest<boost::mp11::mp_int<SizeT>, MemoryUsageT, MemoryTypeT>;

template<class SizeT, class FillFnT, class MemoryUsageT, class MemoryTypeT>
struct FilledMemoryRequest
{
	using Size = SizeT;
	using Buffer = StaticBuffer<Size>;
	using MemoryType = MemoryTypeT;
	using MemoryUsage = MemoryUsageT;
	using FillFn = FillFnT;
};
