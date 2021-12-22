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

#include <cudf/utilities/type_dispatcher.hpp> //< type_to_id<Type>()
#include <cudf/column/column_factories.hpp> //< make_strings_column(...)
#include <cudf/null_mask.hpp> //< create_null_mask(...)
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>

// TODO: DEBUG !!!
#include <boost/core/demangle.hpp> //< boost::core::demangle()
#include <iostream>
#include <meta_json_parser/debug_helpers.h>
#endif

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

template<class T, typename OutputType>
struct CudfNumericColumn {
	using OT = T;

	template<typename TagT, typename ParserOutputDeviceT>
	static void call(const ParserOutputDeviceT& output,
	                 std::vector<std::unique_ptr<cudf::column>> &columns, int i,
					 size_t n_elements, size_t total_size)
	{
		//const uint8_t* data_ptr = output.m_d_outputs[idx++].data().get();
		void* data_ptr = (void *)(output.template Pointer<TagT>());

		std::cout << "converting column " << i << " (numeric: "
				  << boost::core::demangle(typeid(OutputType).name()) << ", "
				  << sizeof(OutputType) << " bytes, "
				  << 8*sizeof(OutputType) << " bits)\n";

		rmm_device_buffer_union u;
		u.data.move_into(data_ptr, total_size); //< data pointer and size in bytes

		auto column = std::make_unique<cudf::column>(
			cudf::data_type{cudf::type_to_id<OutputType>()}, //< The element type
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			u.rmm //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());
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
		void* data_ptr = (void *)(output.template Pointer<TagT>());

		std::cout << "converting column " << i << " (bool)\n";

		rmm_device_buffer_union u;
		u.data.move_into(data_ptr, total_size); //< data pointer and size in bytes

		auto column = std::make_unique<cudf::column>(
			cudf::data_type{cudf::type_id::BOOL8}, //< The element type: boolean using one byte per value
			static_cast<cudf::size_type>(n_elements), //< The number of elements in the column
			u.rmm //< The column's data, as rmm::device_buffer or something convertible
		);

		columns.emplace_back(column.release());
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
		std::cout
			<< "converting column " << i << " (dynamic string: "
			<< n_elements << " elements, " << total_size << " total size)\n";
		std::cout
			<< "- original type is " << boost::core::demangle(typeid(OT).name()) << "\n";

		// - construct child columns
		void* offsets_ptr = (void *)(output.template Pointer<LengthRequestTag>());
		void* strdata_ptr = (void *)(output.template Pointer<DynamicStringRequestTag>());

		rmm_device_buffer_union offsets_u, strdata_u;
		offsets_u.data.move_into(offsets_ptr, n_elements+1);
		strdata_u.data.move_into(strdata_ptr, total_size);

		// DEBUG:
		std::cout
			<< "  - offsets_u.data._size = " << offsets_u.data._size << "\n"
			<< "  - offsets_u.data._capacity = " << offsets_u.data._capacity << "\n"
			<< "  - offsets_u.data._data = " << offsets_u.data._data << "\n"
			<< "  + offsets_u.rmm.size() = " << offsets_u.rmm.size() << "\n"
			<< "  + offsets_u.rmm.capacity() = " << offsets_u.rmm.capacity() << "\n"
			<< "  + offsets_u.rmm.data() = " << offsets_u.rmm.data() << "\n";

		auto offsets_column = std::make_unique<cudf::column>(
			// hopefully cudf::type_id::UINT32 would work as well as cudf::type_id::INT32
			cudf::data_type{cudf::type_to_id<LengthRequestType>()}, //< The element type of offsets
			static_cast<cudf::size_type>(offsets_u.data._size), //< The number of elements in the column
			offsets_u.rmm //< The column's data, as rmm::device_buffer or something convertible
		);
		auto strdata_column = std::make_unique<cudf::column>(
			// NOTE: cudf::type_to_id<char>() returns cudf::type_id::EMPTY, not cudf::type_id::INT8 (???)
			cudf::data_type{cudf::type_id::INT8}, //< The element type of `char`
			static_cast<cudf::size_type>(strdata_u.data._size), //< The number of elements in the column
			strdata_u.rmm //< The column's data, as rmm::device_buffer or something convertible
		);

		// DEBUG:
		std::cout << "- 'offsets_column' is " << boost::core::demangle(typeid(offsets_column).name()) << "\n";
		std::cout << "- 'offsets_column.get()' is " << boost::core::demangle(typeid(offsets_column.get()).name())
		          << " = " << offsets_column.get()
				  << " is " << memory_desc(offsets_column.get())
				  << "\n";
		std::cout << "  - data type = " << type_id_to_name(offsets_column.get()->type().id()) << "\n";
		std::cout << "  - elements in column = " << offsets_column.get()->size() << "\n";
		std::cout << "  - can contain null values = " << (offsets_column.get()->nullable() ? "true" : "false") << "\n";
		std::cout << "  - number of children (directly) = " << offsets_column.get()->num_children()  << "\n";

		std::cout << "- 'strdata_column' is " << boost::core::demangle(typeid(strdata_column).name()) << "\n";
		std::cout << "- 'strdata_column.get()' is " << boost::core::demangle(typeid(strdata_column.get()).name())
		          << " = " << strdata_column.get() << "\n";
		std::cout << "  - data type = " << type_id_to_name(strdata_column.get()->type().id()) << "\n";
		std::cout << "  - elements in column = " << strdata_column.get()->size() << "\n";
		std::cout << "  - can contain null values = " << (strdata_column.get()->nullable() ? "true" : "false") << "\n";
		std::cout << "  - number of children (directly) = " << strdata_column.get()->num_children()  << "\n";

		// - make strings column
		auto column = cudf::make_strings_column(
			n_elements, //< number of elements (number of strings)
			std::move(offsets_column), //< column of offsets into chars data
			std::move(strdata_column), //< column of characters
			0,  //< null count
			//{}, //< null mask
			cudf::create_null_mask(n_elements, cudf::mask_state::UNALLOCATED) //< null mask
		);

		// DEBUG:
		std::cout << "- 'column' is " << boost::core::demangle(typeid(column).name()) << "\n";
		std::cout << "- 'column.get()' is " << boost::core::demangle(typeid(column.get()).name())
		          << " = " << column.get()
				  << " is " << memory_desc(column.get())
				  << "\n";
		std::cout << "  - data type = " << type_id_to_name(column.get()->type().id()) << "\n";
		std::cout << "  - elements in column = " << column.get()->size() << "\n";
		std::cout << "  - can contain null values = " << (column.get()->nullable() ? "true" : "false") << "\n";
		std::cout << "  - number of children (directly) = " << column.get()->num_children()  << "\n";
		auto num_children = column.get()->num_children();
		for (int child_idx = 0; child_idx < num_children; child_idx++) {
			std::cout
				<< "    * child " << child_idx << "\n"
				<< "      - type = " << type_id_to_name(column.get()->child(child_idx).type().id()) << "\n"
				<< "      - size = " << column.get()->child(child_idx).size() << " elements\n";
		}
		std::cout << "getting view: column.get()->view()...\n";
		auto view = column.get()->view();
		std::cout << "- 'column.get()->view() is "
		          << boost::core::demangle(typeid(view).name())
				  << "\n";
		std::cout << "  - number of children (via view) = "
		          << view.num_children()
				  << "\n";

		// - add it to columns
		columns.emplace_back(column.release());
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
