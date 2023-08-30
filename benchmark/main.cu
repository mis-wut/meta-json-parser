#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <iomanip>
#include <chrono>
#include <functional>
#include <map>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <thrust/logical.h>
#include <boost/mp11/set.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>

// used also for benchmarking individual methods
cudaStream_t stream;
// for benchmarking individual functions and methods
cudaEvent_t gpu_beg, gpu_end;

#include <meta_json_parser/parser_output_device.cuh>
#include <meta_json_parser/output_printer.cuh>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jrealnumber.cuh>
#include <meta_json_parser/action/jbool.cuh>
#include <cub/cub.cuh>
#include <CLI/CLI.hpp>

#ifdef HAVE_LIBCUDF
#pragma message("Compiling meta-json-parser-benchmark with HAVE_LIBCUDF")
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/csv.hpp>
#endif /* HAVE_LIBCUDF */

#include <meta_json_parser/debug_helpers.h>
#include <meta_json_parser/checkpoint_results.h>


using namespace boost::mp11;
using namespace std;

template<class Key, int Size>
using StaticCopyFun = JStringStaticCopy<mp_int<Size>, Key>;

template<class Key, int Size>
using DynamicV1CopyFun = JStringDynamicCopy<Key>;

template<class Key, int Size>
using DynamicV2CopyFun = JStringDynamicCopyV2<Key>;

template<class Key, int Size>
using DynamicV3CopyFun = JStringDynamicCopyV3<Key>;

#include "data_def.cuh"
#ifdef HAVE_DTYPES
#pragma message("Included file data_def.cuh has HAVE_DTYPES defined")
#endif /* HAVE_DTYPES */

// NOTE: unix is not an identifier reserved by the Standard.
// However, it is common to find 'unix' #defined on Unix systems
// see https://stackoverflow.com/questions/3770322/is-unix-restricted-keyword-in-c
#ifdef unix
// NOTE: for some reason this still does not work
#undef unix
#endif

enum class workgroup_size { W32, W16, W8, W4 };
enum class end_of_line {
	unknown,
	uniks, //< LF, or "\n": end-of-line convention used by Unix
	win   //< CRLF, or "\r\n": end-of-line convention used by MS Windows
};
enum class dynamic_version {
	v1, //< old version with double copying
	v2, //< new version with single copying
	v3  //< new version with double copying and double buffer
};
enum class dictionary_assumption {
	none, //< no assumptions
	const_order, //< keys in a constant order in JSON
	no_skip, //< do not support skipping over unknown fields
	skip_0, //< skip over unknown fields, with 0-depth stack, i.e. no nesting
	skip_1, //< skip over unknown fields, with 1-depth stack
	skip_default, //< skip over unknown fields
};


// TODO: move to debug_helpers, maybe
const char* workgroup_size_desc(workgroup_size ws)
{
	switch (ws) {
	case workgroup_size::W32:
		return "W32";

	case workgroup_size::W16:
		return "W16";

	case workgroup_size::W8:
		return "W8";

	case workgroup_size::W4:
		return "W4";

	default:
		return "<unknown>";
	};
}

struct benchmark_input
{
	vector<char> data;
	int count;
	workgroup_size wg_size;
	end_of_line eol;
	int bytes_per_string;
};

template<class BaseActionT>
struct benchmark_device_buffers
{
	ParserOutputDevice<BaseActionT> parser_output_buffers;
	char* readonly_buffers;
	char* input_buffer;
	InputIndex* indices_buffer;
	ParsingError* err_buffer;
	void** output_buffers;
	int count;

	std::vector<void*> host_output_buffers;
};

struct cmd_args {
	std::string filename;
#ifdef HAVE_LIBCUDF
	bool use_libcudf_parser;
	bool use_libcudf_writer;
	bool show_cudf_table_structure;
#ifdef HAVE_DTYPES
	// data_def.cuh includes information about dtypes, which we can use
	bool use_dtypes;
#endif /* HAVE_DTYPES */
#endif /* HAVE_LIBCUDF */
	int count;
	workgroup_size wg_size;
	std::string output_csv;
	bool error_check;
	int bytes_per_string;
	dynamic_version version;
	dictionary_assumption dict_assumption;
} g_args;

chrono::high_resolution_clock::time_point cpu_start;
chrono::high_resolution_clock::time_point cpu_stop;
cudaEvent_t gpu_start;
cudaEvent_t gpu_memory_checkpoint;
cudaEvent_t gpu_preprocessing_checkpoint;
cudaEvent_t gpu_parsing_checkpoint;
cudaEvent_t gpu_post_hooks_checkpoint;
cudaEvent_t gpu_output_checkpoint;
cudaEvent_t gpu_error_checkpoint;
#ifdef HAVE_LIBCUDF
cudaEvent_t gpu_convert_checkpoint;
#endif // defined(HAVE_LIBCUDF)
cudaEvent_t gpu_stop;
size_t total_gpu_mem;
size_t free_gpu_mem_start;
size_t free_gpu_mem_stop;
//cudaStream_t stream; // declaration moved earlier

void init_gpu();
void parse_args(int argc, char** argv);
benchmark_input get_input();
end_of_line detect_eol(benchmark_input& input);
void print_results();
void usage();

namespace EndOfLine
{
	struct Unix {};
	struct Win {};
}

template<class EndOfLineT>
struct LineEndingHelper
{
private:
	__device__ __forceinline__ static void error() { assert("Unknown end of line."); }
public:
	__device__ __forceinline__ static uint32_t get_mask(const uint32_t& val) { error(); return 0; }
	__device__ __forceinline__ static bool is_newline(const uint32_t& val) { error(); return false; }
	__device__ __forceinline__ static uint32_t eol_length() { error(); return 0; }
};

template<>
struct LineEndingHelper<EndOfLine::Unix>
{
	__device__ __forceinline__ static uint32_t get_mask(const uint32_t& val)
	{
		return __vcmpeq4(val, '\n\n\n\n');
	}
	__device__ __forceinline__ static bool is_newline(const uint32_t& val)
	{
		return get_mask(val);
	}
	__device__ __forceinline__ static constexpr uint32_t eol_length()
	{
		return 1;
	}
};

/// <summary>
/// Implemented with assumption that \r can only be found right before \n
/// </summary>
template<>
struct LineEndingHelper<EndOfLine::Win>
{
	__device__ __forceinline__ static uint32_t get_mask(const uint32_t& val)
	{
		return __vcmpeq4(val, '\r\r\r\r');
	}
	__device__ __forceinline__ static bool is_newline(const uint32_t& val)
	{
		return get_mask(val);
	}
	__device__ __forceinline__ static constexpr uint32_t eol_length()
	{
		return 2;
	}
};

template<class EndOfLineT>
class OutputIndicesIterator
{
public:

    // Required iterator traits
    typedef OutputIndicesIterator<EndOfLineT>            self_type;              ///< My own type 
    typedef ptrdiff_t                                    difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef cub::KeyValuePair<difference_type, uint32_t> value_type;             ///< The type of the element the iterator can point to
    typedef value_type*                                  pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef value_type                                   reference;              ///< The type of a reference to an element the iterator can point to

#if (THRUST_VERSION >= 100700)
    // Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
    typedef typename thrust::detail::iterator_facade_category<
        thrust::any_system_tag,
        thrust::random_access_traversal_tag,
        value_type,
        reference
      >::type iterator_category;                                        ///< The iterator category
#else
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
#endif  // THRUST_VERSION

private:

    InputIndex*  itr;

public:

    /// Constructor
    __host__ __device__ __forceinline__ OutputIndicesIterator(InputIndex* itr) : itr(itr) {}

	/// Assignment operator
	__device__ __forceinline__ self_type& operator=(const value_type &val)
	{
		int inner_offset = LineEndingHelper<EndOfLineT>::eol_length();
		//undefined behavior for 2 byte jsons. e.g. \n[]\n or \n{}\n
		uint32_t mask = LineEndingHelper<EndOfLineT>::get_mask(val.value);
		switch (mask)
		{
		case 0xFF'00'00'00u:
			inner_offset += 3;
			break;
		case 0x00'FF'00'00u:
			inner_offset += 2;
			break;
		case 0x00'00'FF'00u:
			inner_offset += 1;
			break;
		case 0x00'00'00'FFu:
			//inner_offset += 0;
			break;
		default:
			break;
		}
		*itr = static_cast<InputIndex>(val.key * 4) + inner_offset;
		return *this;
	}

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator[](Distance n)
    {
        self_type offset = OutputIndicesIterator(itr + n);
        return offset;
    }
};

template<class EndOfLineT>
struct IsNewLine
{
	__device__ __forceinline__ bool operator()(const cub::KeyValuePair<ptrdiff_t, uint32_t> c) const {
		return LineEndingHelper<EndOfLineT>::is_newline(c.value);
	}
};

struct NoError
{
	__device__ __host__ bool operator()(ParsingError e)
	{
		return ParsingError::None == e;
	}
};

template<class EndOfLineT>
void find_newlines(char* d_input, size_t input_size, InputIndex* d_indices, int count)
{
	checkpoint_event(gpu_preprocessing_checkpoint, stream, "Finding newlines offsets (indices)");
	InputIndex just_zero = 0;
	cudaMemcpyAsync(d_indices, &just_zero, sizeof(InputIndex), cudaMemcpyHostToDevice, stream);
	
	cub::ArgIndexInputIterator<uint32_t*> arg_iter(reinterpret_cast<uint32_t*>(d_input));
	OutputIndicesIterator<EndOfLineT> out_iter(d_indices + 1); // +1, we need to add 0 at index 0
	
	int* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	int* d_num_selected;
	cudaMalloc(&d_num_selected, sizeof(int));

	cub::DeviceSelect::If(
		d_temp_storage,
		temp_storage_bytes,
		arg_iter,
		out_iter,
		d_num_selected,
		(input_size + 3) / 4,
		IsNewLine<EndOfLineT>(),
		stream
	);

	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cub::DeviceSelect::If(
		d_temp_storage,
		temp_storage_bytes,
		arg_iter,
		out_iter,
		d_num_selected,
		(input_size + 3) / 4,
		IsNewLine<EndOfLineT>(),
		stream
	);

	// Following lines could be commented out as it is only validation step
	cudaStreamSynchronize(stream);
	int h_num_selected = -1;
	cudaMemcpy(&h_num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);
	if (h_num_selected != count)
	{
		cout << "Found " << h_num_selected << " new lines instead of declared " << count << ".\n";
		usage();
	}

	cudaFree(d_temp_storage);
	cudaFree(d_num_selected);
}

template <class BaseActionT, int GroupSizeT>
benchmark_device_buffers<BaseActionT> initialize_buffers(benchmark_input& input, KernelLaunchConfiguration* conf)
{
	using BaseAction = BaseActionT;
  	using GroupSize = mp_int<GroupSizeT>;
  	constexpr int GROUP_SIZE = GroupSizeT;
  	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
  	using GroupCount = mp_int<GROUP_COUNT>;
  	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
  	using PC = ParserConfiguration<RT, BaseAction>;
  	using PK = ParserKernel<PC>;
  	using M3 = typename PK::M3;
  	using BUF = typename M3::ReadOnlyBuffer;
	using KC = typename PK::KC;
	using OM = typename KC::OM;
	constexpr size_t REQUEST_COUNT = boost::mp11::mp_size<typename OutputConfiguration<BaseAction>::RequestList>::value;

	checkpoint_event(gpu_memory_checkpoint, stream, "Memory allocation and copying");
	benchmark_device_buffers<BaseAction> result;
	result.count = input.count;
	result.parser_output_buffers = ParserOutputDevice<BaseAction>(conf, result.count);
	cudaMalloc(&result.readonly_buffers, sizeof(BUF));
	cudaMalloc(&result.input_buffer, input.data.size());
	cudaMalloc(&result.indices_buffer, sizeof(InputIndex) * (input.count + 1));
	cudaMalloc(&result.err_buffer, sizeof(ParsingError) * input.count);
	cudaMalloc(&result.output_buffers, sizeof(void*) * REQUEST_COUNT);

	result.host_output_buffers = vector<void*>(REQUEST_COUNT);
	for (int i = 0; i < REQUEST_COUNT; ++i)
	{
		result.host_output_buffers[i] = result.parser_output_buffers.m_d_outputs[i].data().get();
	}

	cudaMemcpyAsync(result.input_buffer, input.data.data(), input.data.size(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(result.output_buffers, result.host_output_buffers.data(), sizeof(void*) * REQUEST_COUNT, cudaMemcpyHostToDevice, stream);

	//End of line might be passed as an option to the program
	if (input.eol == end_of_line::unknown)
		input.eol = detect_eol(input);

	switch (input.eol)
	{
	case end_of_line::uniks:
		find_newlines<EndOfLine::Unix>
			(result.input_buffer, input.data.size(), result.indices_buffer, input.count);
		break;
	case end_of_line::win:
		find_newlines<EndOfLine::Win>
			(result.input_buffer, input.data.size(), result.indices_buffer, input.count);
		break;
	case end_of_line::unknown:
	default:
		std::cerr << "Unknown end of line character!";
		usage();
		break;
	}

	return result;
}

template<class BaseActionT>
benchmark_device_buffers<BaseActionT> initialize_buffers_dynamic(benchmark_input& input, KernelLaunchConfiguration* conf)
{
	switch (input.wg_size)
	{
	default:
	case workgroup_size::W32:
		return initialize_buffers<BaseActionT, 32>(input, conf);
	case workgroup_size::W16:
		return initialize_buffers<BaseActionT, 16>(input, conf);
	case workgroup_size::W8:
		return initialize_buffers<BaseActionT, 8>(input, conf);
	case workgroup_size::W4:
		return initialize_buffers<BaseActionT, 4>(input, conf);
	}
}

template <class BaseActionT, int GroupSizeT>
void launch_kernel(benchmark_device_buffers<BaseActionT>& device_buffers)
{
	using BaseAction = BaseActionT;
  	using GroupSize = mp_int<GroupSizeT>;
  	constexpr int GROUP_SIZE = GroupSizeT;
  	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
  	using GroupCount = mp_int<GROUP_COUNT>;

  	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
  	using PC = ParserConfiguration<RT, BaseAction>;
  	using PK = ParserKernel<PC>;

	PK pk(device_buffers.parser_output_buffers.m_launch_config, stream);

	pk.Run(
		device_buffers.input_buffer,
		device_buffers.indices_buffer,
		nullptr,
		device_buffers.err_buffer,
		device_buffers.output_buffers,
		device_buffers.count,
		device_buffers.host_output_buffers.data(),
		gpu_parsing_checkpoint,
		gpu_post_hooks_checkpoint
	);
}

template<class BaseActionT>
void launch_kernel_dynamic(benchmark_device_buffers<BaseActionT>& device_buffers, workgroup_size wg_size)
{
	switch (wg_size)
	{
	case workgroup_size::W32:
		return launch_kernel<BaseActionT, 32>(device_buffers);
	case workgroup_size::W16:
		return launch_kernel<BaseActionT, 16>(device_buffers);
	case workgroup_size::W8:
		return launch_kernel<BaseActionT, 8>(device_buffers);
	case workgroup_size::W4:
		return launch_kernel<BaseActionT, 4>(device_buffers);
	default:
		break;
	}
}

template<class BaseActionT>
KernelLaunchConfiguration prepare_dynamic_config(benchmark_input& input)
{
	KernelLaunchConfiguration conf;

	using DynamicStringActions = mp_copy_if_q<
		ActionIterator<BaseActionT>,
		mp_bind<
			mp_similar,
			JStringDynamicCopy<void>,
			_1
		>
	>;

	using DynamicStringActionsV2 = mp_copy_if_q<
		ActionIterator<BaseActionT>,
		mp_bind<
			mp_similar,
			JStringDynamicCopyV2<void>,
			_1
		>
	>;

	using DynamicStringActionsV3 = mp_copy_if_q<
		ActionIterator<BaseActionT>,
		mp_bind<
			mp_similar,
			JStringDynamicCopyV3<void>,
			_1
		>
	>;

	mp_for_each<
		mp_append<
			DynamicStringActions,
			DynamicStringActionsV2
		>
	>([&conf, &input](auto a) {
		using Action = decltype(a);
		using Tag = typename Action::DynamicStringRequestTag;
		conf.SetDynamicSize<BaseActionT, Tag>(input.bytes_per_string);
	});

	mp_for_each<DynamicStringActionsV3>([&conf, &input](auto a) {
		using Action = decltype(a);
		using TagInternal = typename Action::DynamicStringInternalRequestTag;
		conf.SetDynamicSize<BaseActionT, TagInternal>(input.bytes_per_string);
		using Tag = typename Action::DynamicStringRequestTag;
		conf.SetDynamicSize<BaseActionT, Tag>(input.bytes_per_string);
	});

	return std::move(conf);
}

template<class BaseActionT>
ParserOutputHost<BaseActionT> copy_output(benchmark_device_buffers<BaseActionT>& device_buffers)
{
	using BaseAction = BaseActionT;

	checkpoint_event(gpu_output_checkpoint, stream, "Copying output");
	vector<ParsingError> temp_err(device_buffers.count);
	cudaMemcpyAsync(temp_err.data(), device_buffers.err_buffer, sizeof(ParsingError) * device_buffers.count, cudaMemcpyDeviceToHost, stream);
	ParserOutputHost<BaseAction> output = device_buffers.parser_output_buffers.CopyToHost(stream);

	if (g_args.error_check)
	{
		checkpoint_event(gpu_error_checkpoint, stream, "Checking parsing errors");
		bool correct = thrust::all_of(
			thrust::cuda::par.on(stream),
			device_buffers.err_buffer,
			device_buffers.err_buffer + device_buffers.count,
			NoError()
		);
		if (!correct)
		{
			cout << "Parsing errors!\n";
		}
	}
	return output;
}

template<class BaseActionT>
void to_csv(ParserOutputHost<BaseActionT>& output_hosts)
{
	if (g_args.output_csv.empty())
		return;
	cout << "Saving results to '" << g_args.output_csv << "'.\n";
	output_hosts.DropToCsv(g_args.output_csv.c_str());;
}

#ifdef HAVE_LIBCUDF
void to_csv_libcudf(std::string& filename, cudf::io::table_with_metadata const& table_with_metadata);
void to_csv_libcudf(std::string& filename, cudf::table const& cudf_table);

/**
 * This function converts benchmark results on GPU to cuDF format.
 * @param device_buffers represents benchmark results on GPU
 * @return cudf::table @todo: results converted to cuDF compatibile format
 */
template<class BaseActionT>
cudf::table output_to_cudf(benchmark_device_buffers<BaseActionT>& device_buffers)
{
    checkpoint_event(gpu_convert_checkpoint, stream, "Converting to cuDF format");
    return device_buffers.parser_output_buffers.ToCudf(stream);
}
#endif /* defined(HAVE_LIBCUDF) */

template<class BaseActionT>
void main_templated(benchmark_input& input)
{
	cudaMemGetInfo(&free_gpu_mem_start, &total_gpu_mem);
	checkpoint_event(gpu_start, stream, "Initialization");
	KernelLaunchConfiguration conf = prepare_dynamic_config<BaseActionT>(input);
	benchmark_device_buffers<BaseActionT> device_buffers = initialize_buffers_dynamic<BaseActionT>(input, &conf);
	cout << "Workgroup size: " << workgroup_size_desc(input.wg_size)
			<< " [" << static_cast<int>(input.wg_size) << "]\n";
	launch_kernel_dynamic<BaseActionT>(device_buffers, input.wg_size);
	auto host_output = copy_output<BaseActionT>(device_buffers);
#ifdef HAVE_LIBCUDF
    auto cudf_table  = output_to_cudf<BaseActionT>(device_buffers);
#endif // defined(HAVE_LIBCUDF)
	checkpoint_event(gpu_stop, stream, "Total time measured by GPU");
    cudaMemGetInfo(&free_gpu_mem_stop, &total_gpu_mem);
	cudaEventSynchronize(gpu_stop);
	cpu_stop = chrono::high_resolution_clock::now();
	print_results();
#ifndef HAVE_LIBCUDF
    if (!g_args.output_csv.empty())
        to_csv<BaseActionT>(host_output);
#else /* defined(HAVE_LIBCUDF) */
	if (!g_args.output_csv.empty()) {
		if (g_args.use_libcudf_writer) {
			// write CSV file using cudf::io::write_csv()
			to_csv_libcudf(g_args.output_csv, cudf_table);
		} else {
			// write CSV file using host_output.DropToCsv()
			to_csv<BaseActionT>(host_output);
		}
    }
    if (g_args.show_cudf_table_structure)
        describe_table(cudf_table);
#endif /* defined(HAVE_LIBCUDF) */
}

template<template<class, int> class StrFun>
void select_dict_opts(benchmark_input& input) {
	switch (g_args.dict_assumption)
	{
	case dictionary_assumption::none:
		cout << "Assumptions: none\n";
		main_templated<DictCreator<StrFun, mp_list<>>>(input);
		return;
	case dictionary_assumption::const_order:
		cout << "Assumptions: constant order\n";
		main_templated<DictCreator<StrFun, mp_list<
                mp_list<JDictOpts::Order, JDictOpts::Order::ConstOrder>,
                mp_list<JDictOpts::Skip, JDictOpts::Skip::Disable>
        >>>(input);
		return;
	case dictionary_assumption::no_skip:
		cout << "Assumptions: no skipping\n";
		main_templated<DictCreator<StrFun, mp_list<
				// list of options: mp_list<key, value>
				mp_list<JDictOpts::Skip, JDictOpts::Skip::Disable>
			>
		>>(input);
	case dictionary_assumption::skip_0:
		cout << "Assumptions: skipping with depth=0\n";
		main_templated<DictCreator<StrFun, mp_list<
				// list of options: mp_list<key, value>
				mp_list<JDictOpts::Skip, JDictOpts::Skip::Enable_c<0>>
			>
		>>(input);
	case dictionary_assumption::skip_1:
		cout << "Assumptions: skipping with depth=1\n";
		main_templated<DictCreator<StrFun, mp_list<
				// list of options: mp_list<key, value>
				mp_list<JDictOpts::Skip, JDictOpts::Skip::Enable_c<1>>
			>
		>>(input);
	case dictionary_assumption::skip_default:
		cout << "Assumptions: skipping with default depth=8\n";
		main_templated<DictCreator<StrFun, mp_list<
				// list of options: mp_list<key, value>
				mp_list<JDictOpts::Skip, JDictOpts::Skip::Default>
			>
		>>(input);
	default:
		cerr << "Fatal. Unknown dictionary assumption.\n";
		break;
	}
}

void select_string_function(benchmark_input& input) {
	if (input.bytes_per_string == 0)
	{
		cout << "Maximum number of characters per string (built-in): "
		     << STATIC_STRING_SIZE << "\n";
		cout << "Using STATIC string copy.\n";
		select_dict_opts<StaticCopyFun>(input);
	}
	else
	{
		cout << "Maximum number of characters per string (runtime): "
		     << input.bytes_per_string << "\n";
		switch (g_args.version)
		{
		case dynamic_version::v1:
			cout << "Using DYNAMIC V1 string copy.\n";
			select_dict_opts<DynamicV1CopyFun>(input);
			break;
		case dynamic_version::v2:
			cout << "Using DYNAMIC V2 string copy.\n";
			select_dict_opts<DynamicV2CopyFun>(input);
			break;
		case dynamic_version::v3:
			cout << "Using DYNAMIC V3 string copy.\n";
			select_dict_opts<DynamicV3CopyFun>(input);
			break;
		default:
			cerr << "Fatal. Unknown dynamic algorithm chosen.\n";
			break;
		}
	}
}

#ifdef HAVE_LIBCUDF
// forward declarations
void init_libcudf();
cudf::io::json_reader_options prepare_libcudf(benchmark_input& input);
cudf::io::table_with_metadata parse_json_libcudf(cudf::io::json_reader_options const& json_in_opts);
//template<class BaseActionT>
//ParserOutputHost<BaseActionT> copy_output_libcudf(cudf::io::table_with_metadata const& table_with_metadata);
void print_results_libcudf();


/**
 * Parse JSON file using `cudf::io::read_json()` from the libcudf library
 * 
 * The libcudf library is the engine for cuDF, a Pandas-like DataFrame manipulation
 * library for Python that is a part of RAPIDS suite of libraries for data science
 * from NVIDIA.
 * 
 * @attention It assumes that `init_gpu()` - which creates CUDA events,
 * `parse_args()` - which parses command line arguments storing them in global
 * variable `g_args` are run before calling this command, and that the `input`
 * parameter was created with `get_input()`.
 * 
 * @param[in] input  Contains the contents of JSON file as std::vector<char>
 */
void main_libcudf(benchmark_input& input)
{
    init_libcudf();

    cudaMemGetInfo(&free_gpu_mem_start, &total_gpu_mem);
	checkpoint_event(gpu_start, stream, "Initialization");
    auto json_reader_options = prepare_libcudf(input);
#ifdef HAVE_DTYPES
	if (g_args.use_dtypes) {
		// NOTE: dtypes is either std::vector< cudf::data_type >, or std::map< std::string, cudf::data_type >
		std::cout << "dtypes size: " << dtypes.size() << "\n";
		json_reader_options.set_dtypes(dtypes);
	}
#endif /* HAVE_DTYPES */
    auto libcudf_result = parse_json_libcudf(json_reader_options);
    //TODO: auto host_output = copy_output_libcudf(libcudf_result);
	checkpoint_event(gpu_stop, stream, "Total time measured by GPU");
    cudaMemGetInfo(&free_gpu_mem_stop, &total_gpu_mem);
    cudaEventSynchronize(gpu_stop);
    cpu_stop = chrono::high_resolution_clock::now();
    print_results_libcudf();
    if (!g_args.output_csv.empty())
        to_csv_libcudf(g_args.output_csv, libcudf_result);
    if (g_args.show_cudf_table_structure)
        describe_table(libcudf_result);
}
#endif /* defined(HAVE_LIBCUDF) */

int main(int argc, char** argv)
{
	init_gpu();
	parse_args(argc, argv);
	benchmark_input input = get_input();

	// TODO: this hack (turning off buffering for stdout) is a workaround for double free
	// meta-json-parser-benchmark crashes at exit, and its output might not be catched
	// when redirecting it otherwise; alternative would be to flush stdout at the end
	/*
	 terminate called after throwing an instance of 'thrust::system::system_error'
	    what():  CUDA free failed: cudaErrorInvalidValue: invalid argument
	 Aborted (core dumped)
	 */
	setbuf(stdout, NULL);

#ifdef HAVE_LIBCUDF
	if (g_args.use_libcudf_parser) {
		cout << "Using libcudf's cudf::io::read_json\n";
		main_libcudf(input);
	} else {
		cout << "Using meta-JSON-parser\n";
		select_string_function(input);
	}
#else  /* !defined(HAVE_LIBCUDF) */
	select_string_function(input);
#endif /* !defined(HAVE_LIBCUDF) */
	return 0;
}

#ifdef HAVE_LIBCUDF
static CLI::App app{
	"meta-json-parser-benchmark -- benchmark JSON meta-parser, running on GPU\n"
	"(built with support for libcudf library, which is part of RAPIDS.ai)"
#ifdef HAVE_DTYPES
	"\n(data_def.cuh includes information about dtypes for libcudf parser)"
#endif /* HAVE_DTYPES */
};
#else /* !defined(HAVE_LIBCUDF) */
static CLI::App app{"meta-json-parser-benchmark -- benchmark JSON meta-parser, running on GPU"};
#endif /* HAVE_LIBCUDF */

void usage()
{
	// check that it is called after parse_args(),
	// and we have all options and the help message configured
	if (app.parsed())
		exit(app.exit(CLI::CallForHelp()));

	exit(1);
}

void parse_args(int argc, char** argv)
{
	// helpers
	std::map<std::string, workgroup_size> wg_sizes_map{
		{"32", workgroup_size::W32},
        {"16", workgroup_size::W16},
        { "8", workgroup_size::W8},
        { "4", workgroup_size::W4}
	};

	std::map<std::string, dynamic_version> versions_map{
		{"1", dynamic_version::v1},
        {"2", dynamic_version::v2},
        {"3", dynamic_version::v3}
	};

	std::map<bool, dictionary_assumption> const_order_map{
		{false, dictionary_assumption::none},
		{true, dictionary_assumption::const_order}
	};

	std::map<std::string, dictionary_assumption> assumptions_map{
		{"none", dictionary_assumption::none},
		{"const", dictionary_assumption::const_order},
		{"no_skip", dictionary_assumption::no_skip},
		{"skip_0", dictionary_assumption::skip_0},
		{"skip_1", dictionary_assumption::skip_1},
		{"skip", dictionary_assumption::skip_default},
	};

	// defaults
#ifdef HAVE_LIBCUDF
	g_args.use_libcudf_parser = false;
	g_args.use_libcudf_writer = false;
	g_args.show_cudf_table_structure = false;
#ifdef HAVE_DTYPES
	g_args.use_dtypes = false;
#endif /* HAVE_DTYPES */
#endif /* HAVE_LIBCUDF */
	g_args.error_check = false;
	g_args.wg_size = workgroup_size::W32;
	g_args.dict_assumption = dictionary_assumption::none;

	// required parameters as positionals
	app.add_option("JSONLINES_FILE", g_args.filename,
	               "NDJSON / JSONL input file to parse.")
		->required()
		->check(CLI::ExistingFile);
	app.add_option("JSON_COUNT", g_args.count,
	               "Number of lines/objects in the input file.")
		->required()
		->check(CLI::NonNegativeNumber);

	// common options
	app.add_option("-o,--output", g_args.output_csv,
	               "Name for an parsed output CSV file.\n"
                   "If omitted no output is saved.")
		->option_text("CSV_FILENAME");

	// choosing the parser to benchmark, meta-json-parser or libcudf
#ifdef HAVE_LIBCUDF
	auto opt_libcudf_parser =
	app.add_flag("--use-libcudf-parser", g_args.use_libcudf_parser,
	             "Use libcudf JSON parser. Default = false.");
	// NOTE: this matters only when parsing with meta-parser
	// as when using libcudf JSON parser it is used uncodintionally
	app.add_flag("--use-libcudf-writer", g_args.use_libcudf_writer,
	             "Use cudf::io::write_csv to write CSV file also for meta-parser. Default = false.");
	// NOTE: this will be used both for libcudf parser output,
	// and for conversion of metaparser output to cudf::table
	app.add_flag("--cudf-structure", g_args.show_cudf_table_structure,
	             "Describe structure of cudf::table. Default = false.");
#ifdef HAVE_DTYPES
	// NOTE: used only when --use-libcudf-parser is used
	app.add_flag("--use-dtypes", g_args.use_dtypes,
	             "Use data types with libcudf.  Default = false");
#endif /* HAVE_DTYPES */
#endif /* HAVE_LIBCUDF */

	// configuration of meta-json-parser, as options
	auto meta_group = app.add_option_group("meta-json-parser", "meta parser configuration");
	auto opt =
	app.add_option("--ws,--workspace-size", g_args.wg_size,
	               "Workgroup size. Default = 32.")
		->transform(CLI::CheckedTransformer(wg_sizes_map))
		->option_text("32|16|8|4")
		->default_str("32");
	meta_group->add_option(opt);
	opt =
	app.add_flag("-b,--error-checking", g_args.error_check,
	             "Enable error check. If there was a parsing error,\n"
	             "a message will be printed.");
	meta_group->add_option(opt);
	opt =
	app.add_flag("--const-order", g_args.dict_assumption,
				 "Parses json with an assumption of keys in a constant order")
		->transform(CLI::CheckedTransformer(const_order_map));
	meta_group->add_option(opt);
	opt =
	app.add_flag("--assumptions", g_args.dict_assumption,
				 "Parses json with given assumptions")
		->transform(CLI::CheckedTransformer(assumptions_map));
	meta_group->add_option(opt);
	opt =
	app.add_option("-V,--version", g_args.version,
				   "Version of dynamic string parsing.\n"
				   "1 -> old version with double copying. [default]\n"
				   "2 -> new version with single copying.\n"
				   "3 -> new version with double copying and double buffer.")
		->option_text("VERSION")
		->transform(CLI::CheckedTransformer(versions_map))
		->default_str("1");
	meta_group->add_option(opt);
	opt =
	app.add_option("-s,--max-string-size", g_args.bytes_per_string,
	               "Bytes allocated per dynamic string.\n"
	               "For V1: Strings that are too long will be truncated.\n"
	               "For V2/V3: If ammount of needed memory exceeds total\n"
				   "memory allocated program will run into undefined behiavior.\n"
                   "If not provided, then strings with static length will be used.")
		->option_text("BYTES")
		->check(CLI::PositiveNumber);
	meta_group->add_option(opt);

	app.get_formatter()->column_width(40);

	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError &e) {
		exit(app.exit(e));
	}
}

benchmark_input get_input()
{
	ifstream file(g_args.filename, std::ifstream::ate | std::ifstream::binary);
	if (!file.good())
	{
		cout << "Error reading file \"" << g_args.filename << "\".\n";
		usage();
	}
	vector<char> data(file.tellg());
	file.seekg(0);
	//Start measuring CPU time with reading data form disk
	cout << "Reading JSON file: '"  << g_args.filename << "'\n";
	cout << "Expected objects:  " << g_args.count << "\n";
	cpu_start = chrono::high_resolution_clock::now();
	file.read(data.data(), data.size());
	return benchmark_input
	{
		std::move(data),
		g_args.count,
		g_args.wg_size,
		end_of_line::unknown,
		g_args.bytes_per_string
	};
}

void init_gpu()
{
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_memory_checkpoint);
	cudaEventCreate(&gpu_preprocessing_checkpoint);
	cudaEventCreate(&gpu_parsing_checkpoint);
	cudaEventCreate(&gpu_post_hooks_checkpoint);
	cudaEventCreate(&gpu_output_checkpoint);
	cudaEventCreate(&gpu_error_checkpoint);
#ifdef HAVE_LIBCUDF
	cudaEventCreate(&gpu_convert_checkpoint);
#endif // defined(HAVE_LIBCUDF)
	cudaEventCreate(&gpu_stop);
	cudaStreamCreate(&stream);

	// for benchmarking individual functions and methods,
	// created here to avoid putting creation cost in measured unit.
	cudaEventCreate(&gpu_beg);
	cudaEventCreate(&gpu_end);
}

end_of_line detect_eol(benchmark_input& input)
{
	auto found = std::find_if(input.data.begin(), input.data.end(), [](char& c) {
		return c == '\r' || c == '\n';
	});
	if (found == input.data.end())
		return end_of_line::unknown;
	if (*found == '\n')
		return end_of_line::uniks;
	// *found == '\r'
	if ((found + 1) == input.data.end() || *(found + 1) != '\n')
		return end_of_line::unknown;
	return end_of_line::win;
}

void print_results()
{
	float ms;
	int64_t cpu_ns = (cpu_stop - cpu_start).count();
	cudaEventElapsedTime(&ms, gpu_start, gpu_stop);
	int64_t gpu_total = static_cast<int64_t>(ms * 1'000'000.0);

	const int c1 = 40;
	const int c2 = 10;

	print_checkpoint_results();
	cout
		<< setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
		<< setw(c1) << left  << "Total time measured by GPU: "
		<< setw(c2) << right << gpu_total << " ns\n"
		<< setw(c1) << left  << "Total time measured by CPU: "
		<< setw(c2) << right << cpu_ns << " ns\n"
		;
    long int used_gpu_mem = free_gpu_mem_start - free_gpu_mem_stop;
    cout
        << setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
        << setw(c1) << left  << "Total GPU memory: "
        << setw(c2) << right << total_gpu_mem/1024/1024 << " MB\n"
        << setw(c1) << left  << "Free GPU memory at start: "
        << setw(c2) << right << free_gpu_mem_start/1024/1024 << " MB\n"
        << setw(c1) << left  << "Free GPU memory at stop: "
        << setw(c2) << right << free_gpu_mem_stop/1024/1024 << " MB\n"
        << setw(c1) << left  << "Used GPU memory: "
        << setw(c2) << right << used_gpu_mem/1024/1024 << " MB = "
        << setw(6)  << right << fixed << setprecision(2) << 100.0*used_gpu_mem/total_gpu_mem << "% of total\n"
        << "\n"
        << setw(c1) << left  << "Total GPU memory: "
        << setw(12) << right << total_gpu_mem << " bytes\n"
        << setw(c1) << left  << "Used GPU memory: "
        << setw(12) << right << used_gpu_mem << " bytes\n";

	print_checkpoint_events();

	// flush output, just in case
	cout.flush();
}

#ifdef HAVE_LIBCUDF
void init_libcudf()
{
    /*
     * events and streams are created by init_gpu()
     * events needed:
     * - gpu_start
	 * - gpu_memory_checkpoint
     * - gpu_parsing_checkpoint
     * - gpu_stop
     */
}

cudf::io::json_reader_options prepare_libcudf(benchmark_input& input)
{
	checkpoint_event(gpu_memory_checkpoint, stream, "Building input options");

    cudf::io::source_info json_in_info =
        cudf::io::source_info{
            input.data.data(),
            input.data.size()
        };
    cudf::io::json_reader_options json_in_opts =
        cudf::io::json_reader_options::builder(json_in_info)
        .lines(true);

    return json_in_opts;
}

void to_csv_libcudf(std::string& filename, cudf::io::table_with_metadata const& table_with_metadata)
{
	if (filename.empty())
		return;

	cout << "Saving results to '" << filename << "' (via libcudf, with metadata)";
	cout.flush();

	// TODO: make it configurable with respect to style, and if headers are used
	cudf::io::csv_writer_options csv_out_opts =
		cudf::io::csv_writer_options::builder(cudf::io::sink_info{filename},
											  table_with_metadata.tbl->view())
			.inter_column_delimiter(',')
			//.metadata(&table_with_metadata.metadata)  // TODO: fix issue with const-ness type mismatch
			.include_header(false);
	cudf::io::write_csv(csv_out_opts);

	cout << "\n";
}

void to_csv_libcudf(std::string& filename, cudf::table const& cudf_table)
{
	if (filename.empty())
		return;

	cout << "Saving results to '" << filename << "' (via libcudf)";
	cout.flush();

	// TODO: make it configurable with respect to style
	cudf::io::csv_writer_options csv_out_opts =
		cudf::io::csv_writer_options::builder(cudf::io::sink_info{filename},
											  cudf_table.view())
			.inter_column_delimiter(',')
			.include_header(false);
	cudf::io::write_csv(csv_out_opts);

	cout << "\n";
}

cudf::io::table_with_metadata parse_json_libcudf(cudf::io::json_reader_options const& json_in_opts)
{
	checkpoint_event(gpu_parsing_checkpoint, stream, "Parsing json");

    auto result = cudf::io::read_json(json_in_opts);
    return result;
}

void print_results_libcudf()
{
    float ms;
    int64_t cpu_ns = (cpu_stop - cpu_start).count();
    cudaEventElapsedTime(&ms, gpu_start, gpu_stop);
    int64_t gpu_total = static_cast<int64_t>(ms * 1'000'000.0);

    const int c1 = 40; // description width
    const int c2 = 10; // results width

	print_checkpoint_results();
    cout
        << setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
        << setw(c1) << left  << "Total time measured by GPU: "
        << setw(c2) << right << gpu_total << " ns\n"
        << setw(c1) << left  << "Total time measured by CPU: "
        << setw(c2) << right << cpu_ns << " ns\n"
        ;
    // TODO: remove this code duplication (below)
    long int used_gpu_mem = free_gpu_mem_start - free_gpu_mem_stop;
    cout
        << setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
        << setw(c1) << left  << "Total GPU memory: "
        << setw(c2) << right << total_gpu_mem/1024/1024 << " MB\n"
        << setw(c1) << left  << "Free GPU memory at start: "
        << setw(c2) << right << free_gpu_mem_start/1024/1024 << " MB\n"
        << setw(c1) << left  << "Free GPU memory at stop: "
        << setw(c2) << right << free_gpu_mem_stop/1024/1024 << " MB\n"
        << setw(c1) << left  << "Used GPU memory: "
        << setw(c2) << right << used_gpu_mem/1024/1024 << " MB = "
        << setw(6)  << right << fixed << setprecision(2) << 100.0*used_gpu_mem/total_gpu_mem << "% of total\n"
        << "\n"
        << setw(c1) << left  << "Total GPU memory: "
        << setw(12) << right << total_gpu_mem << " bytes\n"
        << setw(c1) << left  << "Used GPU memory: "
        << setw(12) << right << used_gpu_mem << " bytes\n";

	print_checkpoint_events();
}
#endif /* defined(HAVE_LIBCUDF) */
