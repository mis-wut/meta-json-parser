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
#include <cstdint>
#include <type_traits>


#ifdef HAVE_LIBCUDF
#include <vector>
#include <memory>

#include <cudf/types.hpp> // enum class cudf::type_id
#include <cudf/utilities/type_dispatcher.hpp> //< type_to_id<Type>()
#include <cudf/column/column_factories.hpp> //< make_strings_column(...)
#include <cudf/null_mask.hpp> //< create_null_mask(...)
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>

#include <meta_json_parser/strided_range.cuh>
#include <meta_json_parser/action/datetime/datetime_options.h>

// TODO: DEBUG !!!
#include <boost/core/demangle.hpp> //< boost::core::demangle()
// note: boost::core::demangle() is needed in CudfUnknownColumnType, with NDEBUG / without PROFILE_CUDF_CONVERSION
#define PROFILE_CUDF_CONVERSION 1
#ifdef PROFILE_CUDF_CONVERSION
#pragma message("Using PROFILE_CUDF_CONVERSION")
#include <iostream>
#include <chrono>
#include <meta_json_parser/debug_helpers.h>
#include <nvToolsExt.h>  // to help with using profiler
#endif /* defined(PROFILE_CUDF_CONVERSION) */
#endif /* defined(HAVE_LIBCUDF) */


#ifdef HAVE_LIBCUDF
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

// https://github.com/jbenner-radham/libsafec-strnlen_s/blob/master/strnlen_s.h
/**
 * The `my_strnlen_s` function computes the length of the C string pointed to by `s`.
 *
 * Specified in: ISO/IEC TR 24731-1 §6.7.4.3, Programming languages, environments
 * and system software interfaces, Extensions to the C Library, Part I:
 * Bounds-checking interfaces.
 *
 * Modified to be __host__ __device__ function, capable to be run on GPU.
 *
 * @param [in] s       pointer to C string
 * @param [in] maxsize restricted maximum length
 *
 * @return If `s` is a null pointer, then the `my_strnlen_s` function returns zero.
 * Otherwise, the `my_strnlen_s` function returns the number of characters that precede
 * the terminating null character. If there is no null character in the first `maxsize`
 * characters of `s`, then `my_strnlen_s` returns `maxsize`. At most the first `maxsize`
 * characters of `s` shall be accessed by `my_strnlen_s`.
 *
 * @url https://github.com/jbenner-radham/libsafec-strnlen_s
 *
 * @author Bo Berry
 * @author James Benner <james.benner@gmail.com>
 * @copyright The MIT License (MIT)
 *
 * @todo Move to a separate module or at least separate header (?)
 */
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

/**
 * to_str_pair is a function object. Specifically, it is an Adaptable Unary Function.
 *
 * If `f` is an object of class `to_str_pair<maxCharacters>`, and `x` is a character
 * which is a part of collection of C-strings, each of maximum length of `maxCharacters`,
 * then `f(x)` returns `thrust::pair<str, str_length>`, where `str` is C-string beginning
 * at `x`, and `str_length` is its length (up to `maxCharacters`).
 *
 * This operator uses the `my_strlen_s` CPU / GPU function defined earlier.
 *
 * @see unary_function
 *
 * @todo Move to a separate header (this is a template class)
 */
template<int maxCharacters>
struct to_str_pair : public thrust::unary_function<
	const char,
	thrust::pair<const char*, cudf::size_type>
>
{
	using str_pair_t = thrust::pair<const char*, cudf::size_type>;
	using c_str_t = const char*;

	/**
	 * Function call operator. The return value is `thrust::pair<str, str_length>`.
	 */
	__host__ __device__ str_pair_t operator()(const char & x) const
	{
		const char *str = &x;
		const cudf::size_type str_length = my_strnlen_s(str, maxCharacters);
		return std::make_pair(str, str_length);
	}
};

using perf_clock = std::chrono::high_resolution_clock;

template<class T, typename OutputType>
struct CudfNumericColumn {
	using OT = T;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
	                 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
		using OM = typename ParserOutputDeviceT::OM;

#ifdef PROFILE_CUDF_CONVERSION
		nvtxRangePushA("toCudf: numeric");
		std::cout << "converting column " << i << " (numeric: "
				  << boost::core::demangle(typeid(OutputType).name()) << ", "
				  << sizeof(OutputType) << " bytes, "
				  << 8*sizeof(OutputType) << " bits)\n";
		std::cout << "taken from column " << OM::template TagIndex<TagT>::value << "\n";

		perf_clock::time_point cpu_beg, cpu_end;
		cpu_beg = perf_clock::now();
		cudaEventRecord(gpu_beg, stream);
#endif

		//const uint8_t* data_ptr = output.m_d_outputs[idx++].data().get();

		// Create in place object from std::move in order to not trigger destructor
		// after variable ends its scope.
		char buffer[sizeof(thrust::device_vector<uint8_t>) + alignof(thrust::device_vector<uint8_t>)];
		char* aligned_buffer = buffer + alignof(thrust::device_vector<uint8_t>) - reinterpret_cast<intptr_t>(buffer) % alignof(thrust::device_vector<uint8_t>);
		auto* v = new (aligned_buffer) thrust::device_vector<uint8_t>(
			std::move(output.m_d_outputs[OM::template TagIndex<TagT>::value])
		);
		void* data_ptr = v->data().get();

		rmm_device_buffer_union u;
		u.data.move_into(data_ptr, total_size); //< data pointer and size in bytes

		auto column = std::make_unique<cudf::column>(
			cudf::data_type{cudf::type_to_id<OutputType>()}, //< The element type
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			std::move(u.rmm) //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());

#ifdef PROFILE_CUDF_CONVERSION
		cudaEventRecord(gpu_end, stream);
		cpu_end = perf_clock::now();

		int64_t cpu_ns = (cpu_end - cpu_beg).count();
		std::cout << "- time on CPU: " << cpu_ns << " ns\n";

		float ms;
		cudaEventSynchronize(gpu_end);
		cudaEventElapsedTime(&ms, gpu_beg, gpu_end);
		int64_t gpu_ns = static_cast<int64_t>(ms * 1'000'000.0);
		std::cout << "- time on GPU: " << gpu_ns << " ns\n";
		nvtxRangePop();
#endif
	}
};

template<class T>
struct CudfBoolColumn {
	using OT = T;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
	                 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
		using OM = typename ParserOutputDeviceT::OM;

#ifdef PROFILE_CUDF_CONVERSION
		nvtxRangePushA("toCudf: bool");
		std::cout << "converting column " << i << " (bool)\n";
		std::cout << "taken from column " << OM::template TagIndex<TagT>::value << "\n";
		perf_clock::time_point cpu_beg, cpu_end;
		cpu_beg = perf_clock::now();
		cudaEventRecord(gpu_beg, stream);
#endif

		char buffer[sizeof(thrust::device_vector<uint8_t>) + alignof(thrust::device_vector<uint8_t>)];
		char* aligned_buffer = buffer + alignof(thrust::device_vector<uint8_t>) - reinterpret_cast<intptr_t>(buffer) % alignof(thrust::device_vector<uint8_t>);
		auto* v = new (aligned_buffer) thrust::device_vector<uint8_t>(
			std::move(output.m_d_outputs[OM::template TagIndex<TagT>::value])
		);
		void* data_ptr = v->data().get();

		rmm_device_buffer_union u;
		u.data.move_into(data_ptr, total_size); //< data pointer and size in bytes

		auto column = std::make_unique<cudf::column>(
			cudf::data_type{cudf::type_id::BOOL8}, //< The element type: boolean using one byte per value
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			std::move(u.rmm) //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());

#ifdef PROFILE_CUDF_CONVERSION
		cudaEventRecord(gpu_end, stream);
		cpu_end = perf_clock::now();

		int64_t cpu_ns = (cpu_end - cpu_beg).count();
		std::cout << "- time on CPU: " << cpu_ns << " ns\n";

		float ms;
		cudaEventSynchronize(gpu_end);
		cudaEventElapsedTime(&ms, gpu_beg, gpu_end);
		int64_t gpu_ns = static_cast<int64_t>(ms * 1'000'000.0);
		std::cout << "- time on GPU: " << gpu_ns << " ns\n";
		nvtxRangePop();
#endif
	}
};


// TODO: enhance cudf::type_to_id, instead of creating my own
// NOTE: trying to enhance cudf::type_to_id failed for some reason
template <typename T>
inline constexpr cudf::type_id datetype_to_id()
{
	return cudf::type_id::EMPTY;
};

template <>
inline constexpr cudf::type_id datetype_to_id<JDatetimeOptions::TimestampResolution::Seconds>()
{
	return cudf::type_id::TIMESTAMP_SECONDS;
};

template <>
inline constexpr cudf::type_id datetype_to_id<JDatetimeOptions::TimestampResolution::Milliseconds>()
{
	return cudf::type_id::TIMESTAMP_MILLISECONDS;
};


template<class T, typename OutType, typename TimestampType>
struct CudfDatetimeColumn {
	using OT = T;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
	                 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
		using OM = typename ParserOutputDeviceT::OM;

#ifdef PROFILE_CUDF_CONVERSION
		nvtxRangePushA("toCudf: datetime");
		std::cout << "converting column " << i << " (datetime: "
				  << boost::core::demangle(typeid(TimestampType).name()) << " as "
				  << boost::core::demangle(typeid(OutType).name()) << ": "
				  <<   sizeof(OutType) << " bytes, "
				  << 8*sizeof(OutType) << " bits)"
		          << " -> "
				  << type_id_to_name(datetype_to_id<TimestampType>()) << "\n";
		std::cout << "taken from column " << OM::template TagIndex<TagT>::value << "\n";
#endif
		if (!std::is_same_v<OutType,
		                    typename cudf::id_to_type<datetype_to_id<TimestampType>()>::rep>) {
			std::cout << "CudfDatetimeColumn::call(): datetime column output data type " 
			          << boost::core::demangle(typeid(OutType).name())
			          << " do not match required cudf::column data type "
			          << boost::core::demangle(typeid(typename cudf::id_to_type<datetype_to_id<TimestampType>()>::rep).name()) << "\n";
			return;
		}
#ifdef PROFILE_CUDF_CONVERSION
		perf_clock::time_point cpu_beg, cpu_end;
		cpu_beg = perf_clock::now();
		cudaEventRecord(gpu_beg, stream);
#endif

		//const uint8_t* data_ptr = output.m_d_outputs[idx++].data().get();
		char buffer[sizeof(thrust::device_vector<uint8_t>) + alignof(thrust::device_vector<uint8_t>)];
		char* aligned_buffer = buffer + alignof(thrust::device_vector<uint8_t>) - reinterpret_cast<intptr_t>(buffer) % alignof(thrust::device_vector<uint8_t>);
		auto* v = new (aligned_buffer) thrust::device_vector<uint8_t>(
			std::move(output.m_d_outputs[OM::template TagIndex<TagT>::value])
		);
		void* data_ptr = v->data().get();

		rmm_device_buffer_union u;
		u.data.move_into(data_ptr, total_size); //< data pointer and size in bytes

		auto column = std::make_unique<cudf::column>(
			cudf::data_type{datetype_to_id<TimestampType>()}, //< The element type
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			std::move(u.rmm) //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());

#ifdef PROFILE_CUDF_CONVERSION
		cudaEventRecord(gpu_end, stream);
		cpu_end = perf_clock::now();

		int64_t cpu_ns = (cpu_end - cpu_beg).count();
		std::cout << "- time on CPU: " << cpu_ns << " ns\n";

		float ms;
		cudaEventSynchronize(gpu_end);
		cudaEventElapsedTime(&ms, gpu_beg, gpu_end);
		int64_t gpu_ns = static_cast<int64_t>(ms * 1'000'000.0);
		std::cout << "- time on GPU: " << gpu_ns << " ns\n";
		nvtxRangePop();
#endif
	}
};

template<class T, int maxCharacters>
struct CudfStaticStringColumn {
	using OT = T;

	using str_pair_t = thrust::pair<const char*, cudf::size_type>;
	using c_str_t = const char*;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
	                 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
#ifdef PROFILE_CUDF_CONVERSION
		nvtxRangePushA("toCudf: static string");
		std::cout
			<< "converting column " << i
			<< " (static string: "
				<< n_elements << " strings, "
				<< maxCharacters << " max length, "
				<< total_size << " characters"
			<< ")\n";
		perf_clock::time_point cpu_beg, cpu_end;
		cpu_beg = perf_clock::now();
		cudaEventRecord(gpu_beg, stream);
#endif

		void* data_ptr = (void *)(output.template Pointer<TagT>());

		thrust::device_vector<str_pair_t> strings_info(n_elements);
		thrust::device_ptr<const char> char_ptr = thrust::device_pointer_cast(data_ptr);
		auto char_iterator = strided_range(char_ptr, char_ptr + n_elements*maxCharacters, maxCharacters);

		thrust::transform(char_iterator.begin(), char_iterator.end(),
		                  strings_info.begin(),
		                  to_str_pair<maxCharacters>());

		auto column = cudf::make_strings_column(strings_info);

		columns.emplace_back(column.release());

#ifdef PROFILE_CUDF_CONVERSION
		cudaEventRecord(gpu_end, stream);
		cpu_end = perf_clock::now();

		int64_t cpu_ns = (cpu_end - cpu_beg).count();
		std::cout << "- time on CPU: " << cpu_ns << " ns\n";

		float ms;
		cudaEventSynchronize(gpu_end);
		cudaEventElapsedTime(&ms, gpu_beg, gpu_end);
		int64_t gpu_ns = static_cast<int64_t>(ms * 1'000'000.0);
		std::cout << "- time on GPU: " << gpu_ns << " ns\n";
		nvtxRangePop();
#endif
	}
};

template<class T>
struct CudfDynamicStringColumn {
	using OT = T;
	using LengthRequestTag = typename OT::LengthRequestTag;
	using DynamicStringRequestTag = typename OT::DynamicStringRequestTag;
	using LengthRequestType = typename OT::LengthRequest::OutputType;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
	                 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
		using OM = typename ParserOutputDeviceT::OM;

#ifdef PROFILE_CUDF_CONVERSION
		nvtxRangePushA("toCudf: dynamic string");
		std::cout
			<< "converting column " << i << " (dynamic string: "
			<< n_elements << " strings, " << total_size << " characters)\n";
		std::cout
			<< "taken from columns length: " << OM::template TagIndex<LengthRequestTag>::value
			<< " and data: " << OM::template TagIndex<DynamicStringRequestTag>::value
			<< "\n";
		perf_clock::time_point cpu_beg, cpu_end;

		cpu_beg = perf_clock::now();
		cudaEventRecord(gpu_beg, stream);
#endif

		// - construct child columns
		using OM = typename ParserOutputDeviceT::OM;
		char buffer[sizeof(thrust::device_vector<uint8_t>) + alignof(thrust::device_vector<uint8_t>)];
		char* aligned_buffer = buffer + alignof(thrust::device_vector<uint8_t>) - reinterpret_cast<intptr_t>(buffer) % alignof(thrust::device_vector<uint8_t>);

		auto* v = new (aligned_buffer) thrust::device_vector<uint8_t>(
			std::move(output.m_d_outputs[OM::template TagIndex<LengthRequestTag>::value])
		);
		void* offsets_ptr = v->data().get();

		v = new (aligned_buffer) thrust::device_vector<uint8_t>(
			std::move(output.m_d_outputs[OM::template TagIndex<DynamicStringRequestTag>::value])
		);
		void* strdata_ptr = v->data().get();

		rmm_device_buffer_union offsets_u, strdata_u;
		offsets_u.data.move_into(offsets_ptr, n_elements+1);
		strdata_u.data.move_into(strdata_ptr, total_size);

		auto offsets_column = std::make_unique<cudf::column>(
			// hopefully cudf::type_id::UINT32 would work as well as cudf::type_id::INT32
			cudf::data_type{cudf::type_to_id<LengthRequestType>()}, //< The element type of offsets
			static_cast<cudf::size_type>(offsets_u.data._size), //< The number of elements in the column
			std::move(offsets_u.rmm) //< The column's data, as rmm::device_buffer or something convertible
		);
		auto strdata_column = std::make_unique<cudf::column>(
			// NOTE: cudf::type_to_id<char>() returns cudf::type_id::EMPTY, not cudf::type_id::INT8 (???)
			cudf::data_type{cudf::type_id::INT8}, //< The element type of `char`
			static_cast<cudf::size_type>(strdata_u.data._size), //< The number of elements in the column
			std::move(strdata_u.rmm) //< The column's data, as rmm::device_buffer or something convertible
		);

		// - make strings column
		rmm::device_buffer null_mask_no_nulls{}; // default for other overloads of cudf::make_strings_column
#if 0
		auto null_mask_no_nulls = cudf::create_null_mask(n_elements, cudf::mask_state::UNALLOCATED); //< empty null mask, unallocated
#endif
		auto column = cudf::make_strings_column(
			n_elements, //< number of elements (number of strings)
			std::move(offsets_column), //< column of offsets into chars data
			std::move(strdata_column), //< column of characters
			0,  //< null count
			std::move(null_mask_no_nulls) //< null mask for no null values
		);

		// - add it to list of columns to be composed into cudf::table
		columns.emplace_back(column.release());

#ifdef PROFILE_CUDF_CONVERSION
		cudaEventRecord(gpu_end, stream);
		cpu_end = perf_clock::now();

		int64_t cpu_ns = (cpu_end - cpu_beg).count();
		std::cout << "- time on CPU: " << cpu_ns << " ns\n";

		float ms;
		cudaEventSynchronize(gpu_end);
		cudaEventElapsedTime(&ms, gpu_beg, gpu_end);
		int64_t gpu_ns = static_cast<int64_t>(ms * 1'000'000.0);
		std::cout << "- time on GPU: " << gpu_ns << " ns\n";
		nvtxRangePop();
#endif
	}
};

// to be used when we don't know how to convert to cudf::column
template<class T>
struct CudfUnknownColumnType {
	using OT = T;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
					 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
		std::cout
			<< "skipping column " << std::setw(2) << i << " (no convert); "
		    << "n_elements=" << n_elements << "; "
			<< "total_size=" << total_size << "\n"
			<< "- for " << boost::core::demangle(typeid(OT).name()) << "\n";
	}
};

// NOTE: based on HaveOutputRequests, GetOutputRequests and TryGetOutputRequests
template<class T, typename = int>
struct HaveCudfColumnConverter : std::false_type {};

template<class T>
struct HaveCudfColumnConverter<T, decltype(std::declval<typename T::CudfColumnConverter>(), 0)> : std::true_type {};

template<class T>
using GetCudfColumnConverter = typename T::CudfColumnConverter;

template<class T>
using TryGetCudfColumnConverter = boost::mp11::mp_eval_if_not<
	HaveCudfColumnConverter<T>,
	CudfUnknownColumnType<T>, // or `boost::mp11::mp_list<>` for no attempt at conversion
	GetCudfColumnConverter,
	T
>;
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

#ifdef HAVE_LIBCUDF
	// NOTE: patterned on the RequestList from OutputConfiguration in output_manager.cuh
	using CudfColumnConverterList = boost::mp11::mp_flatten<
		boost::mp11::mp_transform<
			TryGetCudfColumnConverter,
			boost::mp11::mp_copy_if<
				ActionIterator<BaseAction>,
				HaveOutputRequests
			>
		>
	>;
#endif

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
		const cudf::size_type n_columns = output_buffers_count;
		std::vector<std::unique_ptr<cudf::column>> columns;
		columns.reserve(n_columns);

		boost::mp11::mp_for_each<typename ParserOutputDevice::CudfColumnConverterList>([&, idx=0](auto k) mutable {
			using CudfConverter = decltype(k);
			using RequestList = typename CudfConverter::OT::OutputRequests;
			using Request = boost::mp11::mp_first<RequestList>;
			using Tag = typename Request::OutputTag;
			using T = typename Request::OutputType;
			const size_t elem_size = OM::template ToAlloc<Tag>(m_launch_config, m_size);

			// DOING: ...
			CudfConverter::template call<Tag>(*this, columns, idx++, m_size, elem_size);
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
