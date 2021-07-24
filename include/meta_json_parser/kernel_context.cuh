#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/config.h>

template<class ParserConfigurationT>
struct KernelContext
{
	using BaseAction = typename ParserConfigurationT::BaseAction;
	using OC = OutputConfiguration<BaseAction>;
	using M3 = MetaMemoryManager<ParserConfigurationT>;
	using OM = OutputManager<BaseAction>;
	using WGR = WorkGroupReader<typename ParserConfigurationT::RuntimeConfiguration::WorkGroupSize>;
	using RT = typename ParserConfigurationT::RuntimeConfiguration;

	M3 m3;
	WGR wgr;
	OM om;

	__device__ __forceinline__ KernelContext(
		typename M3::ReadOnlyBuffer* readonlyBuffers,
		typename M3::SharedBuffers& sharedBuffers,
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
		om(
			output,
			reinterpret_cast<uint32_t*>(
				&m3.template Receive<typename OC::DynamicSizesMemoryRequest>()
			)
		)
	{
	}
};
