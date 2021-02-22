#pragma once
#include "memory_configuration.h"
#include "runtime_configuration.cuh"

template<class RuntimeConfigurationT, class MemoryConfigurationT>
struct ParserConfiguration
{
	using RuntimeConfiguration = RuntimeConfigurationT;
	using MemoryConfiguration = AppendRequest<MemoryConfigurationT, RuntimeConfiguration::MemoryRequest>;
};

