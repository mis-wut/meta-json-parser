#pragma once 
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <boost/mp11/list.hpp>
#include <fstream>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parser_output_host.cuh>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/kernel_launcher.cuh>
#include <meta_json_parser/kernel_launch_configuration.cuh>
#include <cstdint>
#include <type_traits>

#ifdef HAVE_LIBCUDF
#include <vector>
#include <memory>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#endif

struct OutputsPointers
{
	thrust::host_vector<void*> h_outputs;
	thrust::device_vector<void*> d_outputs;
};

template<class BaseActionT>
struct ParserOutputDevice
{
	using BaseAction = BaseActionT;
	using OC = OutputConfiguration<BaseAction>;
	using OM = OutputManager<BaseAction>;

	static constexpr size_t output_buffers_count = boost::mp11::mp_size<typename OC::RequestList>::value;

	size_t m_size;
	const KernelLaunchConfiguration* m_launch_config;
	std::vector<thrust::device_vector<uint8_t>> m_d_outputs;

	ParserOutputDevice() : m_size(0) {}

	ParserOutputDevice(const KernelLaunchConfiguration* launch_config, size_t size)
		: m_size(size), m_launch_config(launch_config), m_d_outputs(output_buffers_count)
	{
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto i) mutable {
			using Request = decltype(i);
			using Tag = typename Request::OutputTag;
			m_d_outputs[idx++] = thrust::device_vector<uint8_t>(
				OM::template ToAlloc<Tag>(m_launch_config, m_size)
			);
		});
	}

	OutputsPointers GetOutputs()
	{
		thrust::host_vector<void*> h_outputs(output_buffers_count);
		auto d_output_it = m_d_outputs.begin();
		for (auto& h_output : h_outputs)
			h_output = d_output_it++->data().get();
		thrust::device_vector<void*> d_outputs(h_outputs);
		return OutputsPointers{
			std::move(h_outputs),
			std::move(d_outputs)
		};
	}

	ParserOutputHost<BaseActionT> CopyToHost(cudaStream_t stream = 0) const
	{
		ParserOutputHost<BaseActionT> result(m_launch_config, m_size);

		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto k) mutable {
			using Request = decltype(k);
			using Tag = typename Request::OutputTag;
			const size_t size = OM::template ToAlloc<Tag>(m_launch_config, m_size);
			if (!OM::template HaveOption<Tag, OutputOptHelpBuffer>())
				cudaMemcpyAsync(result.m_h_outputs[idx].data(), m_d_outputs[idx].data().get(), size, cudaMemcpyDeviceToHost, stream);
			//TODO make result.m_h_outputs depend on OutputOptHelpBuffer and adjust its size instead of skipping elements
			++idx;
		});
		return result;
	}

#ifdef HAVE_LIBCUDF
	/**
	 * This is currently a DUMMY method to convert parser output to cuDF format.
	 *
	 * It will try to avoid new memory allocations and data copying
	 * (device to device) if possible.
	 *
	 * @param stream This is CUDA stream in which operations take place
	 * @return cudf::table which is the data in cuDF format
	 */
	cudf::table ToCudf(cudaStream_t stream = 0) const
	{
		// TODO: return actual conversion result, not an empty table
		// see https://github.com/mis-wut/test-libcudf/blob/main/generate-libcudf.cu
		std::vector<std::unique_ptr<cudf::column>> columns(0);
		return cudf::table(std::move(columns));
	}
#endif /* defined(HAVE_LIBCUDF) */
};
