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

//Source: https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord

template<uint32_t Value>
constexpr __device__ __forceinline__ bool HasZeroByte()
{
	return (((Value)-0x01010101UL) & ~(Value) & 0x80808080UL);
}

constexpr __device__ __forceinline__ bool HasZeroByte(uint32_t value)
{
	return (((value)-0x01010101UL) & ~(value) & 0x80808080UL);
}

//Source: https://graphics.stanford.edu/~seander/bithacks.html#ValueInWord

template<typename Input>
const __device__ __forceinline__ bool HasThisByte(Input value, uint8_t byte)
{
	static_assert(sizeof(Input) == 4, "Input type must have size of 4 bytes");
//#ifdef __CUDA_ARCH__
#if false
	return __vcmpeq4(BytesCast<uint32_t>(value), BytesCast<uint32_t>(uchar4{ byte, byte, byte, byte })) != 0;
#else
	return HasZeroByte (BytesCast<uint32_t>(value) ^ (~0UL / 255 * (byte)));
#endif
}

template<>
const __device__ __forceinline__ bool HasThisByte<uint32_t>(uint32_t value, uint8_t byte)
{
//#ifdef __CUDA_ARCH__
#if false
	return __vcmpeq4(value, BytesCast<uint32_t>(uchar4{ byte, byte, byte, byte })) != 0;
#else
	return HasZeroByte ((value) ^ (~0UL / 255 * (byte)));
#endif
}
