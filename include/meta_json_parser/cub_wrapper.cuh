#pragma once
#include <cub/cub.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda_runtime_api.h>
#include <meta_json_parser/memory_request.h>

struct BitOr
{
	template <typename T>
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
	{
		return a | b;
	}
};

struct BitAnd
{
	template <typename T>
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
	{
		return a & b;
	}
};

template<class OperationT, class WorkGroupSizeT>
struct Reduce_impl
{
	using WarpReduce = typename cub::template WarpReduce<OperationT, WorkGroupSizeT::value>;
	using WarpReduceStorage = typename WarpReduce::TempStorage;

	WarpReduceStorage& storage;
	__device__ INLINE_METHOD Reduce_impl(WarpReduceStorage& _storage) : storage(_storage) {}

	template<class ReduceOperatorT>
	__device__ INLINE_METHOD OperationT Reduce(OperationT value, ReduceOperatorT op)
	{
		return WarpReduce(storage).Reduce(value, op);
	}

	template<class ReduceOperatorT>
	__device__ INLINE_METHOD OperationT Reduce(OperationT value, ReduceOperatorT op, int valid_items)
	{
		return WarpReduce(storage).Reduce(value, op, valid_items);
	}
};

template<class OperationT>
struct ReduceRequestSize {
    template<class RT>
    using fn = boost::mp11::mp_int<
        sizeof(
            typename cub::template WarpReduce<OperationT, RT::WorkGroupSize::value>::TempStorage
        )
    >;
};

template<class OperationT>
using ReduceRequest = MemoryRequestRT_q<
    ReduceRequestSize<OperationT>,
	MemoryUsage::AtomicUsage
>;

template<class OperationT, class WorkGroupSizeT, class KernelContextT>
__device__ INLINE_METHOD Reduce_impl<OperationT, WorkGroupSizeT> Reduce(KernelContextT& kc)
{
	using R = Reduce_impl<OperationT, WorkGroupSizeT>;
	return R(kc.m3
		.template Receive<ReduceRequest<OperationT>>()
		.template Alias<typename R::WarpReduceStorage>()
	);
}

template<class OperationT, class WorkGroupSizeT>
struct Scan_impl
{
	using WarpScan = typename cub::template WarpScan<OperationT, WorkGroupSizeT::value>;
	using WarpScanStorage = typename WarpScan::TempStorage;

	WarpScanStorage& storage;
	__device__ INLINE_METHOD Scan_impl(WarpScanStorage& _storage) : storage(_storage) {}

	__device__ INLINE_METHOD OperationT Broadcast(OperationT value, int sourceThread)
	{
		return WarpScan(storage).Broadcast(value, sourceThread);
	}
};

template<class OperationT>
struct ScanRequestSize {
    template<class RT>
    using fn = boost::mp11::mp_int<
        sizeof(
            typename cub::template WarpScan<OperationT, RT::WorkGroupSize::value>::TempStorage
        )
    >;
};

template<class OperationT>
using ScanRequest = MemoryRequestRT_q<
    ScanRequestSize<OperationT>,
    MemoryUsage::AtomicUsage
>;

template<class OperationT, class WorkGroupSizeT, class KernelContextT>
__device__ INLINE_METHOD Scan_impl<OperationT, WorkGroupSizeT> Scan(KernelContextT& kc)
{
	using R = Scan_impl<OperationT, WorkGroupSizeT>;
	return R(kc.m3
		.template Receive<ScanRequest<OperationT>>()
		.template Alias<typename R::WarpScanStorage>()
	);
}

