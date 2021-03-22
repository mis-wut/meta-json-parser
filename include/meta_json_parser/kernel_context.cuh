#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/config.h>

template<class ParserConfigurationT, class OutputConfigurationT>
struct KernelContext
{
	using M3 = MetaMemoryManager<ParserConfigurationT>;
	using OM = OutputManager<OutputConfigurationT>;
	using WGR = WorkGroupReader<ParserConfigurationT::RuntimeConfiguration::WorkGroupSize>;
	using RT = ParserConfigurationT::RuntimeConfiguration;

	M3 m3;
	WGR wgr;
	OM om;

	__device__ __forceinline__ KernelContext(
		M3::ReadOnlyBuffer* readonlyBuffers,
		M3::SharedBuffers& sharedBuffers,
		const char* input,
		const InputIndex* indices,
		void** output
	) :
		m3(sharedBuffers, readonlyBuffers),
		wgr(
			m3.template Receive<typename WGR::MemoryRequest>(),
			input + indices[RT::InputId()],
			input + indices[RT::InputId() + 1]
		),
		om(output)
	{
	}
};