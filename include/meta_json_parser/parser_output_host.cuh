#pragma once 
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <boost/mp11/list.hpp>
#include <fstream>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/kernel_launcher.cuh>
#include <meta_json_parser/output_printer.cuh>
#include <cstdint>
#include <type_traits>

template<class BaseActionT>
struct ParserOutputHost
{
	using BaseAction = BaseActionT;
	using OC = OutputConfiguration<BaseAction>;
	using OM = OutputManager<BaseAction>;

	static constexpr size_t output_buffers_count = boost::mp11::mp_size<typename OC::RequestList>::value;

	size_t m_size;
	const KernelLaunchConfiguration* m_launch_config;
	std::vector<thrust::host_vector<uint8_t>> m_h_outputs;

	ParserOutputHost() : m_size(0) {}

	ParserOutputHost(const KernelLaunchConfiguration* launch_config, size_t size)
		: m_size(size), m_launch_config(launch_config), m_h_outputs(output_buffers_count)
	{
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto i) mutable {
			using Request = decltype(i);
			using Tag = typename Request::OutputTag;
			if (!OM::template HaveOption<Tag, OutputOptHelpBuffer>())
				m_h_outputs[idx] = thrust::host_vector<uint8_t>(
					OM::template ToAlloc<Tag>(m_launch_config, m_size)
				);
			++idx;
		});
	}

	template<class OutputTagT>
	void* Pointer()
	{
		return m_h_outputs[OM::template TagIndex<OutputTagT>::value].data();
	}

	template<class OutputTagT>
	void const* Pointer() const
	{
		return m_h_outputs[OM::template TagIndex<OutputTagT>::value].data();
	}

	void DropToCsv(const char* filename) const
	{
		std::ofstream csv(filename);
		for (auto i = 0ull; i < m_size; ++i)
		{
			using PrintableActions = boost::mp11::mp_copy_if<
				ActionIterator<BaseAction>,
				HaveOutputRequests
			>;
			boost::mp11::mp_for_each<PrintableActions> ([&, idx=0](auto a) mutable {
				using Action = decltype(a);
				using Printer = GetPrinter<Action>;
				if (idx != 0)
					csv << ',';
				Printer::Print(*this, i, csv);
				++idx;
			});
			csv << '\n';
		}
	}
};
