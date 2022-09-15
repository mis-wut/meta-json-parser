#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/function.hpp>
#include <meta_json_parser/meta_math.h>
#include <meta_json_parser/work_group_reader.cuh>

template<class WorkGroupSizeT, class WorkGroupCountT>
struct RuntimeConfiguration
{
	using WorkGroupSize = WorkGroupSizeT;
	using WorkGroupCount = WorkGroupCountT;
	using BlockDimX = WorkGroupSize;
	using BlockDimY = WorkGroupCount;
	using BlockDimZ = boost::mp11::mp_int<1>;
	using BlockSize = boost::mp11::mp_mul<WorkGroupSize, WorkGroupCount>;
	static_assert(
		boost::mp11::mp_less_equal<
			BlockSize,
			boost::mp11::mp_int<1024>
		>::value,
		"Workers count cannot be greater than maximum block size"
	);
	using MemoryRequest = typename WorkGroupReader<WorkGroupSize>::MemoryRequest;
	static dim3 BlockDim()
	{
		return { BlockDimX::value, BlockDimY::value, BlockDimZ::value };
	}
	__device__ __forceinline__ static unsigned int BlockId()
	{
		return blockIdx.x;
	}
	__device__ __forceinline__ static unsigned int GroupInBlockId()
	{
		return threadIdx.y;
	}
	__device__ __forceinline__ static unsigned int WorkersInBlock()
	{
		return WorkGroupSize::value + WorkGroupCount::value;
	}
	__device__ __forceinline__ static unsigned int WorkerInBlockId()
	{
		return GroupInBlockId() * WorkGroupSize::value + WorkerId();
	}
	__device__ __forceinline__ static unsigned int WorkerId()
	{
		return threadIdx.x;
	}
	__device__ __forceinline__ static unsigned int GroupSize()
	{
		return WorkGroupSize::value;
	}
	/// <summary>
	/// Group id within block
	/// </summary>
	/// <returns></returns>
	__device__ __forceinline__ static unsigned int LocalGroupId()
	{
		return threadIdx.y;
	}
	__device__ __forceinline__ static unsigned int WarpGroupId()
	{
		return threadIdx.y % (32 / GroupSize());
	}
	__device__ __forceinline__ static unsigned int GlobalGroupId()
	{
		return threadIdx.y + blockIdx.x * WorkGroupCount::value;
	}
	__device__ __forceinline__ static unsigned int InputId()
	{
		return GlobalGroupId();
	}
    __device__ __forceinline__ static bool IsLeader() {
	    return WorkerId() == 0;
	}
};
