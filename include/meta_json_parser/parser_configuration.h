#pragma once
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>

template<class RuntimeConfigurationT, class MemoryConfigurationT>
struct ParserConfiguration
{
	using RuntimeConfiguration = RuntimeConfigurationT;
	using MemoryConfiguration = AppendRequest<MemoryConfigurationT, RuntimeConfiguration::MemoryRequest>;
};

