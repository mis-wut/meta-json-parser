#pragma once
#include <device_launch_parameters.h>

template<typename TDestination, typename TSource>
__host__ __device__ __forceinline__ TDestination& BytesCast(TSource& source)
{
	return *reinterpret_cast<TDestination*>(&source);
}

template<typename TDestination, typename TSource>
__host__ __device__ __forceinline__ TDestination BytesCast(TSource&& source)
{
	return *reinterpret_cast<TDestination*>(&source);
}

template<typename BytesOffsetT, typename InT>
__host__ __device__ __forceinline__ char* OffsetBytes(InT& obj)
{
	return reinterpret_cast<char*>(&obj) + BytesOffsetT::value;
}

template<typename BytesOffsetT, typename OutT, typename InT>
__host__ __device__ __forceinline__ OutT& OffsetBytesAs(InT& obj)
{
	return *reinterpret_cast<OutT*>(reinterpret_cast<char*>(&obj) + BytesOffsetT::value);
}
