#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cub/util_type.cuh>
#include <boost/mp11/map.hpp>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/byte_algorithms.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/meta_math.h>
#include <meta_json_parser/memory_request.h>

template<typename WorkingGroupSizeT>
struct WorkGroupReaderBase
{
	static constexpr int GROUP_SIZE = WorkingGroupSizeT::value;
	static constexpr std::size_t MEMORY_ALIGNMENT = 16;
	static_assert(IsPower2_c<MEMORY_ALIGNMENT>::type::value, "MEMORY_ALIGNMENT must be power of 2");
	static constexpr std::size_t ALIGNMENT_MASK = MEMORY_ALIGNMENT - 1;
	static constexpr std::size_t BUFFER_COUNT = 2;
	static_assert(BUFFER_COUNT == 2, "Algorithms are fixed for 2 buffers.");
	using VectorType = uint32_t;
	static constexpr std::size_t BUFFER_SIZE = sizeof(VectorType) * GROUP_SIZE;
	static_assert(IsPower2_c<BUFFER_SIZE>::type::value, "BUFFER_SIZE must be power of 2.");
	using MemoryRequest = MemoryRequest_c<BUFFER_COUNT * BUFFER_SIZE, MemoryUsage::ActionUsage, MemoryType::Shared>;

private:
#pragma nv_exec_check_disable
	template<int WorkGroupSize>
	static __device__ __forceinline__ uint32_t __detail_ballot_sync(int predicate)
	{
		constexpr uint32_t GROUPS_IN_WARP = 32 / WorkGroupSize;
		constexpr uint32_t MASK = 0xFF'FF'FF'FFu >> (32 - WorkGroupSize);
		uint32_t result = __ballot_sync(0xFF'FF'FF'FFu, predicate);
		return (result >> ((threadIdx.y % GROUPS_IN_WARP) * WorkGroupSize)) & MASK;
	}

	template<>
	static __device__ __forceinline__ uint32_t __detail_ballot_sync<32>(int predicate)
	{
		return __ballot_sync(0xFF'FF'FF'FFu, predicate);
	}

#pragma nv_exec_check_disable
	template<int WorkGroupSize>
	static __device__ __forceinline__ uint32_t __detail_all_sync(int predicate)
	{
		constexpr uint32_t MASK = 0xFF'FF'FF'FFu >> (32 - WorkGroupSize);
		return __detail_ballot_sync<WorkGroupSize>(predicate) == MASK;
	}

	template<>
	static __device__ __forceinline__ uint32_t __detail_all_sync<32>(int predicate)
	{
		return __all_sync(0xFF'FF'FF'FFu, predicate);
	}
public:
#pragma nv_exec_check_disable
	__device__ __forceinline__ uint32_t ballot_sync(int predicate)
	{
		return __detail_ballot_sync<GROUP_SIZE>(predicate);
	}

#pragma nv_exec_check_disable
	__device__ __forceinline__ uint32_t all_sync(int predicate)
	{
		return __detail_all_sync<GROUP_SIZE>(predicate);
	}
};

template<typename WorkingGroupSizeT>
struct WorkGroupReader : public WorkGroupReaderBase<WorkingGroupSizeT>
{
	using Base = WorkGroupReaderBase<WorkingGroupSizeT>;
	static constexpr int GROUP_SIZE = Base::GROUP_SIZE;
	static constexpr std::size_t MEMORY_ALIGNMENT = Base::MEMORY_ALIGNMENT;
	static constexpr std::size_t ALIGNMENT_MASK = Base::ALIGNMENT_MASK;
	static constexpr std::size_t BUFFER_COUNT = Base::BUFFER_COUNT;
	using VectorType = Base::VectorType;
	static constexpr std::size_t BUFFER_SIZE = Base::BUFFER_SIZE;
	using MemoryRequest = Base::MemoryRequest;
protected:
	const char* mSource;
	const char* mEndSource;
	char* mBufferPtr;
	int mInBufferOffset;
	uint8_t mWarpId;

	__device__ __noinline__ VectorType LoadLastBytes()
	{
		VectorType payload = VectorType(0);
		if (mSource != mEndSource)
		{
			const char* beginCopy = mSource + ThreadIndex() * sizeof(VectorType);
			const char* endCopy = beginCopy + sizeof(VectorType);
			if (endCopy <= mEndSource)
				payload = reinterpret_cast<const VectorType*>(mSource)[ThreadIndex()];
			//finish from 1 to 3 chars
			if (beginCopy < mEndSource && mEndSource < endCopy)
			{
				reinterpret_cast<char4*>(&payload)->x = mSource[4 * ThreadIndex()];
				if (beginCopy + 1 < mEndSource)
					reinterpret_cast<char4*>(&payload)->y = mSource[4 * ThreadIndex() + 1];
				if (beginCopy + 2 < mEndSource)
					reinterpret_cast<char4*>(&payload)->z = mSource[4 * ThreadIndex() + 2];
			}
			mSource = mEndSource;
		}
		return payload;
	}

public:
	__device__ INLINE_METHOD void Load(unsigned int bufferId)
	{
		VectorType payload;
		if (mSource + BUFFER_SIZE < mEndSource)
		{
			payload = reinterpret_cast<const VectorType*>(mSource)[ThreadIndex()];
			mSource += BUFFER_SIZE;
		}
		else if (mSource == mEndSource) 
		{
			//Clear buffer
			payload = VectorType(0);
		}
		else
		{
			payload = LoadLastBytes();
		}
		reinterpret_cast<VectorType*>(mBufferPtr + BUFFER_SIZE * bufferId)[ThreadIndex()] = payload;
		__syncwarp();
	}

public:

	__device__ INLINE_METHOD unsigned int CurrentBufferId() const
	{
		return mInBufferOffset / BUFFER_SIZE;
	}

	__device__ INLINE_METHOD unsigned int ThreadIndex() const { return threadIdx.x; }

	__device__ INLINE_METHOD char CurrentChar()
	{
		return mBufferPtr[(mInBufferOffset + ThreadIndex()) % (BUFFER_COUNT * BUFFER_SIZE)];
	}

	__device__ INLINE_METHOD char PeekChar(int thread_id)
	{
		return mBufferPtr[(mInBufferOffset + thread_id) % (BUFFER_COUNT * BUFFER_SIZE)];
	}

	//forward cannot exceed BUFFER_SIZE / 2
	//Broken with prefetch
	__device__ INLINE_METHOD char PeekForward(int thread_id, int forward)
	{
#if _DEBUG
		assert(forward <= BUFFER_SIZE / 2);
#endif
		return mBufferPtr[(mInBufferOffset + thread_id + forward) % (BUFFER_COUNT * BUFFER_SIZE)];
	}

	__device__ INLINE_METHOD void AdvanceBy(int advance)
	{
#ifdef _DEBUG
		assert(advance <= GROUP_SIZE);
#endif
		const unsigned int preActiveBuffer = CurrentBufferId();
		mInBufferOffset += advance;
		if (mInBufferOffset >= BUFFER_COUNT * BUFFER_SIZE)
			mInBufferOffset -= BUFFER_COUNT * BUFFER_SIZE;
		const unsigned int postActiveBuffer = CurrentBufferId();
		if (preActiveBuffer != postActiveBuffer)
		{
			Load(postActiveBuffer == (BUFFER_COUNT - 1) ? 0 : (postActiveBuffer + 1));
		}
	}

	__device__ INLINE_METHOD WorkGroupReader(
		MemoryRequest::Buffer& pBuffers,
		const char* pSource,
		const char* pEndSource = nullptr) :
		mSource(pSource),
		mEndSource(pEndSource == nullptr ? reinterpret_cast<char*>(~0x0ull) : pEndSource),
		mWarpId(threadIdx.y),
		mBufferPtr(reinterpret_cast<char*>(pBuffers.data)),
		mInBufferOffset(0)
	{
#ifdef _DEBUG
		assert(blockDim.x == GROUP_SIZE);
		assert(blockDim.z == 1);
#endif
		uint32_t missAlignment = BytesCast<uint32_t>(mSource) & 0x3u;
		if (missAlignment)
		{
			//Clear buffer
			reinterpret_cast<VectorType*>(mBufferPtr + (BUFFER_COUNT - 1) * BUFFER_SIZE)[ThreadIndex()] = VectorType(0);
			missAlignment = 4 - missAlignment;
			mInBufferOffset = BUFFER_COUNT * BUFFER_SIZE - missAlignment;
			if (threadIdx.x < missAlignment && mSource + threadIdx.x < mEndSource)
				 mBufferPtr[mInBufferOffset + threadIdx.x] = mSource[threadIdx.x];
			mSource += missAlignment;
			if (mSource > mEndSource)
				mSource = mEndSource;
			Load(0);
		}
		else
		{
			Load(0);
			Load(1);
		}
	}
};

template<typename WorkingGroupSizeT>
struct WorkGroupReaderPrefetch : public WorkGroupReaderBase<WorkingGroupSizeT>
{
	using Base = WorkGroupReaderBase<WorkingGroupSizeT>;
	static constexpr int GROUP_SIZE = Base::GROUP_SIZE;
	static constexpr std::size_t MEMORY_ALIGNMENT = Base::MEMORY_ALIGNMENT;
	static constexpr std::size_t ALIGNMENT_MASK = Base::ALIGNMENT_MASK;
	static constexpr std::size_t BUFFER_COUNT = Base::BUFFER_COUNT;
	using VectorType = Base::VectorType;
	static constexpr std::size_t BUFFER_SIZE = Base::BUFFER_SIZE;
	using MemoryRequest = Base::MemoryRequest;
	//TODO current implementation is inefficient. VectorType should always be 4 bytes.
	//Different strategies of loading need to be implemented for different group sizes
protected:
	const char* mSource;
	const char* mEndSource;
	char* mBufferPtr;
	int mInBufferOffset;
	uint8_t mWarpId;
	VectorType mPrefetch;
	
	__device__ INLINE_METHOD void LoadPrefetch()
	{
		if (mSource + BUFFER_SIZE < mEndSource)
		{
			mPrefetch = reinterpret_cast<const VectorType*>(mSource)[ThreadIndex()];
			mSource += BUFFER_SIZE;
		}
		else 
		{
			//Clear buffer
			mPrefetch = VectorType(0);
			if (mSource != mEndSource)
			{
				const char* beginCopy = mSource + ThreadIndex() * sizeof(VectorType);
				const char* endCopy = beginCopy + sizeof(VectorType);
				if (endCopy <= mEndSource)
					mPrefetch = reinterpret_cast<const VectorType*>(mSource)[ThreadIndex()];
				//finish from 1 to 3 chars
				if (beginCopy < mEndSource && mEndSource < endCopy)
				{
					reinterpret_cast<char4*>(&mPrefetch)->x = mSource[4 * ThreadIndex()];
					if (beginCopy + 1 < mEndSource)
						reinterpret_cast<char4*>(&mPrefetch)->y = mSource[4 * ThreadIndex() + 1];
					if (beginCopy + 2 < mEndSource)
						reinterpret_cast<char4*>(&mPrefetch)->z = mSource[4 * ThreadIndex() + 2];
				}
				mSource = mEndSource;
			}
		}
	}

	__device__ INLINE_METHOD void StorePrefetch(unsigned int bufferId)
	{
		reinterpret_cast<VectorType*>(mBufferPtr + BUFFER_SIZE * bufferId)[ThreadIndex()] = mPrefetch;
		__syncwarp();
	}

public:

	__device__ INLINE_METHOD unsigned int CurrentBufferId() const
	{
		return mInBufferOffset / BUFFER_SIZE;
	}

	__device__ INLINE_METHOD unsigned int ThreadIndex() const { return threadIdx.x; }

	__device__ INLINE_METHOD char CurrentChar()
	{
		return mBufferPtr[(mInBufferOffset + ThreadIndex()) % (BUFFER_COUNT * BUFFER_SIZE)];
	}

	__device__ INLINE_METHOD char PeekChar(int thread_id)
	{
		return mBufferPtr[(mInBufferOffset + thread_id) % (BUFFER_COUNT * BUFFER_SIZE)];
	}

	//forward cannot exceed BUFFER_SIZE / 2
	//Broken with prefetch
	__device__ INLINE_METHOD char PeekForward(int thread_id, int forward)
	{
#if _DEBUG
		assert(forward <= BUFFER_SIZE / 2);
#endif
		return mBufferPtr[(mInBufferOffset + thread_id + forward) % (BUFFER_COUNT * BUFFER_SIZE)];
	}

	__device__ INLINE_METHOD void AdvanceBy(int advance)
	{
#ifdef _DEBUG
		assert(advance <= GROUP_SIZE);
#endif
		const unsigned int preActiveBuffer = CurrentBufferId();
		mInBufferOffset += advance;
		if (mInBufferOffset >= BUFFER_COUNT * BUFFER_SIZE)
			mInBufferOffset -= BUFFER_COUNT * BUFFER_SIZE;
		const unsigned int postActiveBuffer = CurrentBufferId();
		if (preActiveBuffer != postActiveBuffer)
		{
			StorePrefetch(postActiveBuffer == (BUFFER_COUNT - 1) ? 0 : (postActiveBuffer + 1));
			LoadPrefetch();
		}
	}

	__device__ INLINE_METHOD WorkGroupReaderPrefetch(
		MemoryRequest::Buffer& pBuffers,
		const char* pSource,
		const char* pEndSource = nullptr) :
		mSource(pSource),
		mEndSource(pEndSource == nullptr ? reinterpret_cast<char*>(~0x0ull) : pEndSource),
		mWarpId(threadIdx.y),
		mBufferPtr(reinterpret_cast<char*>(pBuffers.data)),
		mInBufferOffset(0)
	{
#ifdef _DEBUG
		assert(blockDim.x == GROUP_SIZE);
		assert(blockDim.z == 1);
#endif
		uint32_t missAlignment = BytesCast<uint32_t>(mSource) & 0x3u;
		if (missAlignment)
		{
			//Clear buffer
			reinterpret_cast<VectorType*>(mBufferPtr + (BUFFER_COUNT - 1) * BUFFER_SIZE)[ThreadIndex()] = VectorType(0);
			missAlignment = 4 - missAlignment;
			mInBufferOffset = BUFFER_COUNT * BUFFER_SIZE - missAlignment;
			if (threadIdx.x < missAlignment && mSource + threadIdx.x < mEndSource)
				 mBufferPtr[mInBufferOffset + threadIdx.x] = mSource[threadIdx.x];
			mSource += missAlignment;
			if (mSource > mEndSource)
				mSource = mEndSource;
			LoadPrefetch();
			StorePrefetch(0);
			LoadPrefetch();
		}
		else
		{
			LoadPrefetch();
			StorePrefetch(0);
			LoadPrefetch();
			StorePrefetch(1);
			LoadPrefetch();
		}
	}
};
