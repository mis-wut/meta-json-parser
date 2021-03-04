#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/intelisense_silencer.h>

//Forward declaration
template<class KernelLauncherT, class ...KernelArgsT>
struct KernelLauncherCudaArgs;

template<class ...KernelArgsT>
struct KernelLauncher
{
	typedef void(*KernelFunc)(KernelArgsT...);
	KernelFunc mKernel;
	inline KernelLauncher(KernelFunc pKernel) : mKernel(pKernel) { }
	inline KernelLauncherCudaArgs<KernelLauncher, KernelArgsT...> operator()(dim3 pBlocks, dim3 pThreads,
		int pSharedMemory = 0, cudaStream_t pStream = 0) const
	{
		return KernelLauncherCudaArgs<KernelLauncher, KernelArgsT...>(*this, pBlocks, pThreads, pSharedMemory, pStream);
	}
};

template<class BlockDimXT, class BlockDimYT, class BlockDimZT, class SharedMemoryT, class ...KernelArgsT>
struct KernelLauncherFixedResources
{
	typedef void(*KernelFunc)(KernelArgsT...);
	KernelFunc mKernel;
	inline KernelLauncherFixedResources(KernelFunc pKernel) : mKernel(pKernel) { }
	inline KernelLauncherCudaArgs<KernelLauncherFixedResources, KernelArgsT...> operator()(dim3 pBlocks, cudaStream_t pStream = 0) const
	{
		dim3 threads = { BlockDimXT::value, BlockDimYT::value, BlockDimZT::value };
		int sharedMemory = SharedMemoryT::value;
		return KernelLauncherCudaArgs<KernelLauncherFixedResources, KernelArgsT...>(*this, pBlocks, threads, sharedMemory, pStream);
	}
};

template<class ...KernelArgsT>
inline KernelLauncher<KernelArgsT...> Launch(void(*pKernel)(KernelArgsT...))
{
	return KernelLauncher<KernelArgsT...>(pKernel);
}

template<class KernelLauncherT, class ...KernelArgsT>
struct KernelLauncherCudaArgs
{
	KernelLauncherT mLauncher;
	dim3 mBlocks;
	dim3 mThreads;
	int mSharedMemory;
	cudaStream_t mStream;
	inline KernelLauncherCudaArgs(const KernelLauncherT& pLauncher, dim3 pBlocks, dim3 pThreads, int pSharedMemory = 0, cudaStream_t pStream = 0) :
		mLauncher(pLauncher), mBlocks(pBlocks), mThreads(pThreads), mSharedMemory(pSharedMemory), mStream(pStream) { }

	inline void operator()(KernelArgsT... args)
	{
		//TODO start here. On release mKernel is nullptr
		mLauncher.mKernel CU_LAUNCH(mBlocks, mThreads, mSharedMemory, mStream)(args...);
	}
};

