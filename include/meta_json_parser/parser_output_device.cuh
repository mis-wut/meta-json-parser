#pragma once 
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
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
#include <meta_json_parser/strided_range.cuh>
#include <cstdint>
#include <type_traits>


// TODO: DEBUG !!!
#include <boost/core/demangle.hpp>
#include <iostream>

// TODO: make configurable with CMake
#define HAVE_LIBCUDF
#if defined(HAVE_LIBCUDF)
#include <cudf/utilities/type_dispatcher.hpp> //< type_to_id<Type>()
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>

struct rmm_device_buffer_data {
	// NOTE: Horrible, horrible hack needed because of design decisions of rmm::device_buffer
	// idea of the hack borrowed from https://stackoverflow.com/a/19209874/46058

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
		std::cout << "converting column " << i << " (numeric: "
				  << boost::core::demangle(typeid(OutputType).name()) << ", "
				  << elem_size << " bytes, "
				  << 8*elem_size << " bits"
				  << ")\n";

		rmm_device_buffer_union u;
		rmm_device_buffer_data buffer = u.data;
		buffer.move_into(data_ptr, elem_size * n_elements); //< data pointer and size in bytes

		auto column = std::make_unique<cudf::column>(
			cudf::data_type{cudf::type_to_id<OutputType>()}, //< The element type
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			u.rmm //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());
	}
};

struct CudfBoolColumn {
	static void call(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 void *data_ptr, size_t n_elements, size_t elem_size)
	{
		std::cout << "converting column " << i << " (bool)\n";

		// TODO: fix code repetition
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

// https://github.com/jbenner-radham/libsafec-strnlen_s/blob/master/strnlen_s.h
__host__ __device__
size_t my_strnlen_s(const char *s, size_t maxsize)
{
	if (s == NULL)
		return 0;

	size_t count = 0;
	while (*s++ && maxsize--) {
		count++;
	}

	return count;
}

template<int maxCharacters>
struct to_str_pair : public thrust::unary_function<
	const char,
	thrust::pair<const char*, cudf::size_type>
>
{
	using str_pair_t = thrust::pair<const char*, cudf::size_type>;
	using c_str_t = const char*;

	__host__ __device__ str_pair_t operator()(const char & x) const
	{
		const char *str = &x;
		const cudf::size_type str_length = my_strnlen_s(str, maxCharacters);
		return std::make_pair(str, str_length);
	}
};

template<int maxCharacters>
struct CudfStringColumnFromStaticMemory {
	using str_pair_t = thrust::pair<const char*, cudf::size_type>;
	using c_str_t = const char*;

	static void call(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 void *data_ptr, size_t n_elements, size_t elem_size)
	{
		std::cout << "converting column " << i << " (string, static memory,"
				  << " max length=" << maxCharacters << ","
				  << " elem_size=" << elem_size
				  << ")\n";

		thrust::device_vector<str_pair_t> strings_info(n_elements);
		thrust::device_ptr<const char> char_ptr = thrust::device_pointer_cast(data_ptr);
		auto char_iterator = strided_range(char_ptr, char_ptr + n_elements*maxCharacters, maxCharacters);

		thrust::transform(char_iterator.begin(), char_iterator.end(),
						  strings_info.begin(),
						  to_str_pair<maxCharacters>());

		auto column = cudf::make_strings_column(strings_info);

		columns.emplace_back(column.release());
	}
};

// to be used when we don't know how to convert to cudf::column
struct CudfUnknownColumnType {
	static void call(std::vector<std::unique_ptr<cudf::column>> &columns, int i,
	                 void *data_ptr, size_t n_elements, size_t elem_size)
	{
		std::cout << "skipping column " << i << " (don't know how to convert to cudf::column)\n";
	}
};

// TODO: might be not needed, we might want to call CudfConverter::call directly
// NOTE: there is no partial specialization for functions

// generic, requires CudfConverter type to have static `call` method
// REMOVED 



template<class T, typename = int>
struct HaveCudfConverter : std::false_type {};

template<class T>
struct HaveCudfConverter<T, decltype(std::declval<typename T::CudfConverter>(), 0)> : std::true_type {};

template<class T>
using GetCudfConverter = typename T::CudfConverter;

template<class T>
using TryGetCudfConverter = boost::mp11::mp_eval_if_not<
	HaveCudfConverter<T>,
	CudfUnknownColumnType,
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

#if defined(HAVE_LIBCUDF)
	using CudfColumnsConverterList = boost::mp11::mp_flatten<
		boost::mp11::mp_transform<
			TryGetCudfConverterX,
			ActionIterator<BaseAction>
		>
	>;
#endif /* defined(HAVE_LIBCUDF) */

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
		boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0, dynamic_idx=0](auto k) mutable {
			using Request = decltype(k);
			using Tag = typename Request::OutputTag;
			const size_t size = OM::template ToAlloc<Tag>(m_launch_config, m_size);
			cudaMemcpyAsync(result.m_h_outputs[idx].data(), m_d_outputs[idx].data().get(), size, cudaMemcpyDeviceToHost, stream);
			++idx;
		});
		return result;
	}

	// TODO: remove code duplication wrt parser_output_host.cuh
	// NOTE: using `uint8_t*` instead of `void*`.
	template<class TagT>
	void* Pointer()
	{
		return m_d_outputs[OM::template TagIndex<TagT>::value].data().get();
	}

	template<class TagT>
	void const* Pointer() const
	{
		return m_d_outputs[OM::template TagIndex<TagT>::value].data().get();
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
			CudfConverter::call<Tag>(*this, columns, idx-1, m_size, elem_size);
			#if 0
			std::cout << "- k is " << boost::core::demangle(typeid(k).name()) << "\n";
			std::cout << "- T is " << boost::core::demangle(typeid(typename Request::OutputType).name()) << "\n";
			#endif
		});

		// create a table (which will be turned into DataFrame equivalent)
		std::cout << "created table...\n";
		cudf::table table{std::move(columns)}; // std::move or std::forward
		std::cout << "...with " << table.num_columns() << " / " << n_columns
				  << " columns and " << table.num_rows() << " rows\n";

		return table;
	}
#endif /* defined(HAVE_LIBCUDF) */
};
