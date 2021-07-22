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

// TODO: DEBUG !!!
#include <boost/core/demangle.hpp>
#include <iostream>

// TODO: make configurable with CMake
#define HAVE_LIBCUDF
#if defined(HAVE_LIBCUDF)
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>

struct rmm_device_buffer_data {
	// NOTE: Horrible, horrible hack needed because of design decisions of rmm::device_buffer
	// borrowed from https://stackoverflow.com/a/19209874/46058

	// NOTE: the types and order of fields copied from <rmm/device_buffer.hpp>, must be the same!
	void* _data{nullptr};
	std::size_t _size{};
	std::size_t _capacity{};
	rmm::cuda_stream_view _stream{};
	rmm::mr::device_memory_resource* _mr{rmm::mr::get_current_device_resource()};

	void move_into(void * device_ptr, std::size_t size_bytes)
	{
		_data = device_ptr;
		_size = _capacity = size_bytes;
	}
};

union rmm_device_buffer_union {
	rmm::device_buffer rmm;
	rmm_device_buffer_data data;

	rmm_device_buffer_union() : rmm() {}
	~rmm_device_buffer_union() {}
};

template<typename OutputType>
struct CudfNumericColumn {
	static void call(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 void *data_ptr, size_t n_elements, size_t elem_size)
	{
		std::cout << "skipping column " << i << " (numeric: "
				  << boost::core::demangle(typeid(OutputType).name())
				  << ")\n";
	}
};

struct CudfBoolColumn {
	static void call(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 void *data_ptr, size_t n_elements, size_t elem_size)
	{
		std::cout << "converting column " << i << " (bool)\n";

		rmm_device_buffer_union u;
		rmm_device_buffer_data buffer = u.data;
		buffer.move_into(data_ptr, elem_size * n_elements); //< data pointer and size in bytes
		auto column = std::make_unique<cudf::column>(
			cudf::data_type{cudf::type_id::BOOL8}, //< The element type: boolean using one byte per value, 0 == false, else true.
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			u.rmm //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());
	}
};

template<int maxCharacters>
struct CudfStringColumnFromStaticMemory {
	static void call(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 void *data_ptr, size_t n_elements, size_t elem_size)
	{
		std::cout << "skipping column " << i << " (string, static memory, max length=" << maxCharacters << ")\n";
	}
};

// TODO: enhance, specialize
// NOTE: there is no partial specialization for functions
// https://stackoverflow.com/questions/8061456/c-function-template-partial-specialization
// using the "layer of indirection" solution, which is a good idea anyway

// generic, requires CudfConverter type to have static `call` method
template<typename CudfConverter>
void add_column(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
				void *data_ptr, size_t n_elements, size_t elem_size)
{
	CudfConverter::call(columns, i, data_ptr, n_elements, elem_size);
}


// specialization, for when we don't know how to convert to cudf::column
template<>
void add_column<std::false_type>(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
								 void *data_ptr, size_t n_elements, size_t elem_size)
{
	std::cout << "skipping column " << i << " (don't know how to convert to cudf::column)\n";
}




template<class T, typename = int>
struct HaveCudfConverter : std::false_type {};

template<class T>
struct HaveCudfConverter<T, decltype(std::declval<typename T::CudfConverter>(), 0)> : std::true_type {};

template<class T>
using GetCudfConverter = typename T::CudfConverter;

template<class T>
using TryGetCudfConverter = boost::mp11::mp_eval_if_not<
	HaveCudfConverter<T>,
	std::false_type,
	GetCudfConverter,
	T
>;

// TODO: replace this stub implementation
// TODO: better name
template<class T>
using GetCudfConverterX = std::pair<
	TryGetCudfConverter<T>,
	boost::mp11::mp_first<GetOutputRequests<T>>
>;

// TODO: better name
template<class T>
using TryGetCudfConverterX = boost::mp11::mp_eval_if_not<
	HaveOutputRequests<T>,
	boost::mp11::mp_list<>,
	GetCudfConverterX,
	T
>;
#endif /* HAVE_LIBCUDF */

template<class BaseActionT>
struct ParserOutputDevice
{
	using OC = OutputConfiguration<BaseActionT>;
	using OM = OutputManager<OC>;

#if defined(HAVE_LIBCUDF)
	using BaseAction = BaseActionT;
	using CudfColumnsConverterList = boost::mp11::mp_flatten<
		boost::mp11::mp_transform<
			TryGetCudfConverterX,
			ActionIterator<BaseAction>
		>
	>;
	/* TODO: does not work
	using TypedRequestList = boost::mp11::mp_transform<
		boost::mp11::mp_list,
		CudfColumnsConverterList,
		OC::RequestList
	>;
	 */
#endif /* defined(HAVE_LIBCUDF) */

	static constexpr size_t output_buffers_count = boost::mp11::mp_size<typename OC::RequestList>::value;

	size_t m_size;
	const KernelLaunchConfiguration* m_launch_config;
	thrust::device_vector<uint8_t> m_d_outputs[output_buffers_count];

	ParserOutputDevice() : m_size(0) {}

	ParserOutputDevice(const KernelLaunchConfiguration* launch_config, size_t size)
		: m_size(size), m_launch_config(launch_config)
	{
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto i) mutable {
			using Request = decltype(i);
			using Tag = typename Request::OutputTag;
			m_d_outputs[idx++] = thrust::device_vector<uint8_t>(OM::template ToAlloc<Tag>(m_launch_config, m_size));
		});
	}

	ParserOutputHost<BaseActionT> CopyToHost(cudaStream_t stream = 0) const
	{
		ParserOutputHost<BaseActionT> result(m_launch_config, m_size);
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0, dynamic_idx=0](auto k) mutable {
			using Request = decltype(k);
			const size_t size = m_size * (
				IsTemplate<DynamicOutputRequest>::template fn<Request>::value
				? m_launch_config->dynamic_sizes[dynamic_idx++]
				: sizeof(typename Request::OutputType)
				);
			cudaMemcpyAsync(result.m_h_outputs[idx].data(), m_d_outputs[idx].data().get(), size, cudaMemcpyDeviceToHost, stream);
			++idx;
		});
		return result;
	}

#if defined(HAVE_LIBCUDF)
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
		cudf::size_type n_columns = boost::mp11::mp_size<typename OC::RequestList>::value;
		std::vector<std::unique_ptr<cudf::column>> columns;
		columns.reserve(n_columns);

		boost::mp11::mp_for_each<typename ParserOutputDevice::CudfColumnsConverterList>([&, idx=0, dynamic_idx=0](auto k) mutable {
			using CudfConverter = typename decltype(k)::first_type;

			using Request = typename decltype(k)::second_type;
			using Tag = typename Request::OutputTag;
			using T = typename Request::OutputType;

			const uint8_t* ptr = m_d_outputs[idx++].data().get();
			const T* cast_ptr = reinterpret_cast<const T*>(ptr);

			const size_t elem_size = (
					IsTemplate<DynamicOutputRequest>::template fn<Request>::value
					? m_launch_config->dynamic_sizes[dynamic_idx++]
					: sizeof(typename Request::OutputType)
			);
			const size_t size = m_size * elem_size;

			// DOING: ...
			add_column<CudfConverter>(columns, idx-1, (void *)ptr, m_size, elem_size);
			#if 0
			std::cout << "- k is " << boost::core::demangle(typeid(k).name()) << "\n";
			std::cout << "- T is " << boost::core::demangle(typeid(typename Request::OutputType).name()) << "\n";
			#endif
		});

		// create a table (which will be turned into DataFrame equivalent)
		std::cout << "created table...\n";
		cudf::table table{std::move(columns)}; // std::move or std::forward
		std::cout << "...with " << table.num_columns() << " columns and " << table.num_rows() << " rows\n";

		return table;
	}
#endif /* defined(HAVE_LIBCUDF) */
};
