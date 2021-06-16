#pragma once
#include <cstdint>
#include <vector>

struct KernelLaunchConfiguration
{
	std::vector<uint32_t> dynamic_sizes;
};

