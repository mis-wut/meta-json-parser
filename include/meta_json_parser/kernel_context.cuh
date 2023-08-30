#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_requirements.cuh>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/action_iterator.h>
#include <meta_json_parser/config.h>

template<class ParserConfigurationT>
struct KernelContext
{
	using type = KernelContext<ParserConfigurationT>;
	using ParserConfiguration = ParserConfigurationT;
	using PC = ParserConfiguration;
	using BaseAction = typename ParserConfigurationT::BaseAction;
	using OC = OutputConfiguration<BaseAction>;
	using M3 = MetaMemoryManager<ParserConfigurationT>;
	using OM = OutputManager<BaseAction>;
	using RT = typename ParserConfigurationT::RuntimeConfiguration;

	using ParserRequirements =
	boost::mp11::mp_flatten<
		boost::mp11::mp_transform<
			GetParserRequirements,
			boost::mp11::mp_copy_if<
				ActionIterator<BaseAction>,
				HaveParserRequirements
			>
		>
	>;

	using KeepDistance = boost::mp11::mp_not_equal<
		boost::mp11::mp_find<
			ParserRequirements,
			ParserRequirement::KeepDistance
		>,
	    boost::mp11::mp_size<ParserRequirements>
	>;

	using WGR = WorkGroupReader<
		typename RT::WorkGroupSize,
		KeepDistance
	>;

	M3 m3;
	WGR wgr;
	OM om;

	__device__ __forceinline__ KernelContext(
		typename M3::ReadOnlyBuffer* readonlyBuffers,
		typename M3::SharedBuffers& sharedBuffers,
		const char* input,
		const InputIndex* indices,
		const int* indices_positions,
		void** output,
		const uint32_t count
	) :
		m3(sharedBuffers, readonlyBuffers),
		wgr(
			m3.template Receive<typename WGR::MemoryRequest>(),
			RT::InputId() < count ? input + (indices_positions == nullptr ? indices[RT::InputId()] : indices[indices_positions[RT::InputId()]]) : nullptr,
			RT::InputId() < count ? input + (indices_positions == nullptr ? indices[RT::InputId() + 1]: indices[indices_positions[RT::InputId()] + 1]) : nullptr
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
