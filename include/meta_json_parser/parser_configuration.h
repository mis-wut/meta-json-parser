#pragma once
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>

template<class RuntimeConfigurationT, class BaseActionT, class AdditionalMemoryRequestsT = boost::mp11::mp_list<>>
struct ParserConfiguration
{
	using BaseAction = BaseActionT;
	using RuntimeConfiguration = RuntimeConfigurationT;
	using OutputConfiguration = OutputConfiguration<BaseActionT>;
	using MemoryConfiguration = MemoryConfiguration<
		BaseActionT,
		boost::mp11::mp_append<
			boost::mp11::mp_push_front<
				typename OutputConfiguration::MemoryRequests,
				RuntimeConfiguration::MemoryRequest
			>,
			AdditionalMemoryRequestsT
		>
	>;
};

