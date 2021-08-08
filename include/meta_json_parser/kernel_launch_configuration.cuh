#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

template<class BaseActionT>
struct OutputManager;

struct KernelLaunchConfiguration
{
	std::vector<uint32_t> dynamic_sizes;

	template<class BaseActionT, class TagT>
	void SetDynamicSize(uint32_t size)
	{
		constexpr size_t idx = OutputManager<BaseActionT>::template DynamicTagIndex<TagT>::value;
		static_assert(
			idx < boost::mp11::mp_size<typename OutputManager<BaseActionT>::DynamicOnlyRequestList>::value,
			"Provided tag doesn't exists in dynamic requests list."
		);
		if (dynamic_sizes.size() <= idx)
		{
			const size_t old_len = dynamic_sizes.size();
			dynamic_sizes.resize(idx + 1);
			std::fill(dynamic_sizes.begin() + old_len, dynamic_sizes.end(), 0ull);
		}
		dynamic_sizes[idx] = size;
	}
};

