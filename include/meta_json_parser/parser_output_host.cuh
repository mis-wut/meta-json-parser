#pragma once 
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <boost/mp11/list.hpp>
#include <fstream>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/kernel_launcher.cuh>
#include <cstdint>
#include <type_traits>

template<class BaseActionT>
struct ParserOutputHost
{
	using OC = OutputConfiguration<BaseActionT>;
	using OM = OutputManager<OC>;

	size_t m_size;
	const KernelLaunchConfiguration* m_launch_config;
	thrust::host_vector<uint8_t> m_h_outputs[boost::mp11::mp_size<typename OC::RequestList>::value];

	ParserOutputHost() : m_size(0) {}

	ParserOutputHost(const KernelLaunchConfiguration* launch_config, size_t size)
		: m_size(size), m_launch_config(launch_config)
	{
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0, dynamic_idx=0](auto i) mutable {
			using Request = decltype(i);
			if (IsTemplate<DynamicOutputRequest>::template fn<Request>::value)
			{
				m_h_outputs[idx++] = thrust::host_vector<uint8_t>(m_size * m_launch_config->dynamic_sizes[dynamic_idx++]);
			}
			else
			{
				m_h_outputs[idx++] = thrust::host_vector<uint8_t>(m_size * sizeof(typename Request::OutputType));
			}
		});
	}

	template<class OutputTag>
	void*& Pointer()
	{
		return m_h_outputs[OM::template TagIndex<OutputTag>::value].data();
	}

	void DropToCsv(const char* filename) const
	{
		std::ofstream csv(filename);
		for (auto i = 0ull; i < m_size; ++i)
		{
			boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto k) mutable {
				using Request = decltype(k);
				using T = typename Request::OutputType;
				if (idx != 0)
					csv << ',';
				const uint8_t* ptr = m_h_outputs[idx++].data();
				const T* cast_ptr = reinterpret_cast<const T*>(ptr);
				csv << cast_ptr[i];
			});
			csv << '\n';
		}
	}
};
