#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/config.h>

template<class ParserConfigurationT>
struct KernelContext
{
	using M3 = MetaMemoryManager<ParserConfigurationT>;
	using WGR = WorkGroupReader<ParserConfigurationT::RuntimeConfiguration::WorkGroupSize>;
	using RT = ParserConfigurationT::RuntimeConfiguration;

	M3 m3;
	WGR wgr;

	__device__ __forceinline__ KernelContext(M3::SharedBuffers& sharedBuffers, const char* input, const InputIndex* indices) :
		m3(sharedBuffers),
		wgr(
			m3.template Receive<typename WGR::MemoryRequest>(),
			input + indices[RT::InputId()],
			input + indices[RT::InputId() + 1]
		)
	{
	}
};