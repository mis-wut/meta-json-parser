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
#include <cstdint>
#include <type_traits>

// TODO: make configurable with CMake
#define HAVE_LIBCUDF
#if defined(HAVE_LIBCUDF)
#include <cudf/table/table.hpp>
#endif /* HAVE_LIBCUDF */


template<class BaseActionT>
struct ParserOutputDevice
{
	using OC = OutputConfiguration<typename BaseActionT::OutputRequests>;
	using OM = OutputManager<OC>;

	size_t m_size;
	thrust::device_vector<uint8_t> m_d_outputs[boost::mp11::mp_size<typename OC::RequestList>::value];

	ParserOutputDevice() : m_size(0) {}

	ParserOutputDevice(size_t size) : m_size(size)
	{
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto i) mutable {
			using Request = decltype(i);
			m_d_outputs[idx++] = thrust::device_vector<uint8_t>(m_size * sizeof(typename Request::OutputType));
		});
	}

	ParserOutputHost<BaseActionT> CopyToHost(cudaStream_t stream = 0) const
	{
		ParserOutputHost<BaseActionT> result(m_size);
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto k) mutable {
			using Request = decltype(k);
			const size_t size = m_size * sizeof(typename Request::OutputType);
			cudaMemcpyAsync(result.m_h_outputs[idx].data(), m_d_outputs[idx].data().get(), size, cudaMemcpyDeviceToHost, stream);
			++idx;
		});
		return result;
	}

#if defined(HAVE_LIBCUDF)
    /**
     * This is a DUMMY method to convert parser output to cuDF format.
     *
     * It will try to avoid new memory allocations and data copying
     * (device to device) if possible.
     *
     * @param stream This is CUDA stream in which operations take place
     * @return cudf::table which is the data in cuDF format
     */
    cudf::table ToCudf(cudaStream_t stream = 0) const
    {
        cudf::size_type n_columns = boost::mp11::mp_size<typename OC::RequestList>::value;
        std::vector<std::unique_ptr<cudf::column>> columns(n_columns);

        boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto k) mutable {
            using Request = decltype(k);
            using Tag = typename Request::OutputTag;
            using T = typename Request::OutputType;
            const uint8_t* ptr = m_d_outputs[idx].data().get();
            const T* cast_ptr = reinterpret_cast<const T*>(ptr);

            // TODO: ...
		});

        
        return cudf::table();
    }
#endif /*!defined(HAVE_LIBCUDF) */
};
