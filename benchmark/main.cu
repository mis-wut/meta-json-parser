#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <iomanip>
#include <chrono>
#include <functional>
#include <cuda_runtime_api.h>
#include <thrust/logical.h>
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
#include <meta_json_parser/action/jbool.cuh>
#include <cub/cub.cuh>

// TODO: make configurable with CMake
#define HAVE_LIBCUDF
#if defined(HAVE_LIBCUDF)
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/csv.hpp>
#endif /* HAVE_LIBCUDF */


using namespace boost::mp11;
using namespace std;

// KEYS
using K_L1_date = mp_string<'d', 'a', 't', 'e'>;
using K_L1_lat = mp_string<'l', 'a', 't'>;
using K_L1_lon = mp_string<'l', 'o', 'n'>;
using K_L1_is_checked = mp_string<'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_name = mp_string<'n', 'a', 'm', 'e'>;
using K_L1_1_date = mp_string<'1', '_', 'd', 'a', 't', 'e'>;
using K_L1_1_lat = mp_string<'1', '_', 'l', 'a', 't'>;
using K_L1_1_lon = mp_string<'1', '_', 'l', 'o', 'n'>;
using K_L1_1_is_checked = mp_string<'1', '_', 'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_1_name = mp_string<'1', '_', 'n', 'a', 'm', 'e'>;
using K_L1_2_date = mp_string<'2', '_', 'd', 'a', 't', 'e'>;
using K_L1_2_lat = mp_string<'2', '_', 'l', 'a', 't'>;
using K_L1_2_lon = mp_string<'2', '_', 'l', 'o', 'n'>;
using K_L1_2_is_checked = mp_string<'2', '_', 'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_2_name = mp_string<'2', '_', 'n', 'a', 'm', 'e'>;
using K_L1_3_date = mp_string<'3', '_', 'd', 'a', 't', 'e'>;
using K_L1_3_lat = mp_string<'3', '_', 'l', 'a', 't'>;
using K_L1_3_lon = mp_string<'3', '_', 'l', 'o', 'n'>;
using K_L1_3_is_checked = mp_string<'3', '_', 'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_3_name = mp_string<'3', '_', 'n', 'a', 'm', 'e'>;

// DICT
using BaseAction = JDict < mp_list <
	mp_list<K_L1_date, JStringStaticCopy<mp_int<32>, K_L1_date>>,
	mp_list<K_L1_lat, JNumber<uint32_t, K_L1_lat>>,
	mp_list<K_L1_lon, JNumber<uint32_t, K_L1_lon>>,
	mp_list<K_L1_is_checked, JBool<uint8_t, K_L1_is_checked>>,
	mp_list<K_L1_name, JStringStaticCopy<mp_int<32>, K_L1_name>>,
	mp_list<K_L1_1_date, JStringStaticCopy<mp_int<32>, K_L1_1_date>>,
	mp_list<K_L1_1_lat, JNumber<uint32_t, K_L1_1_lat>>,
	mp_list<K_L1_1_lon, JNumber<uint32_t, K_L1_1_lon>>,
	mp_list<K_L1_1_is_checked, JBool<uint8_t, K_L1_1_is_checked>>,
	mp_list<K_L1_1_name, JStringStaticCopy<mp_int<32>, K_L1_1_name>>,
	mp_list<K_L1_2_date, JStringStaticCopy<mp_int<32>, K_L1_2_date>>,
	mp_list<K_L1_2_lat, JNumber<uint32_t, K_L1_2_lat>>,
	mp_list<K_L1_2_lon, JNumber<uint32_t, K_L1_2_lon>>,
	mp_list<K_L1_2_is_checked, JBool<uint8_t, K_L1_2_is_checked>>,
	mp_list<K_L1_2_name, JStringStaticCopy<mp_int<32>, K_L1_2_name>>,
	mp_list<K_L1_3_date, JStringStaticCopy<mp_int<32>, K_L1_3_date>>,
	mp_list<K_L1_3_lat, JNumber<uint32_t, K_L1_3_lat>>,
	mp_list<K_L1_3_lon, JNumber<uint32_t, K_L1_3_lon>>,
	mp_list<K_L1_3_is_checked, JBool<uint8_t, K_L1_3_is_checked>>,
	mp_list<K_L1_3_name, JStringStaticCopy<mp_int<32>, K_L1_3_name>>
>> ;

using PrinterMap = mp_list<
	mp_list<K_L1_is_checked, mp_quote<BoolPrinter>>,
	mp_list<K_L1_1_is_checked, mp_quote<BoolPrinter>>,
	mp_list<K_L1_2_is_checked, mp_quote<BoolPrinter>>,
	mp_list<K_L1_3_is_checked, mp_quote<BoolPrinter>>
>;

constexpr int CHARS_PER_JSTRING = 32;

enum workgroup_size { W32, W16, W8 };

struct benchmark_input
{
	vector<char> data;
	int count;
	workgroup_size wg_size;
};

struct benchmark_device_buffers
{
	ParserOutputDevice<BaseAction> parser_output_buffers;
	char* readonly_buffers;
	char* input_buffer;
	InputIndex* indices_buffer;
	ParsingError* err_buffer;
	void** output_buffers;
	int count;
};

struct cmd_args {
	std::string filename;
	int count;
	workgroup_size wg_size;
	std::string output_csv;
	bool error_check;
} g_args;

chrono::high_resolution_clock::time_point cpu_start;
chrono::high_resolution_clock::time_point cpu_stop;
cudaEvent_t gpu_start;
cudaEvent_t gpu_memory_checkpoint;
cudaEvent_t gpu_preprocessing_checkpoint;
cudaEvent_t gpu_parsing_checkpoint;
cudaEvent_t gpu_output_checkpoint;
cudaEvent_t gpu_error_checkpoint;
#if defined(HAVE_LIBCUDF)
cudaEvent_t gpu_convert_checkpoint;
#endif // defined(HAVE_LIBCUDF)
cudaEvent_t gpu_stop;
cudaStream_t stream;

void init_gpu();
void parse_args(int argc, char** argv);
benchmark_input get_input();
benchmark_device_buffers initialize_buffers_dynamic(benchmark_input& input);
void launch_kernel_dynamic(benchmark_device_buffers& device_buffers, workgroup_size wg_size);
void find_newlines(char* d_input, size_t input_size, InputIndex* d_indices, int count);
ParserOutputHost<BaseAction> copy_output(benchmark_device_buffers& device_buffers);
#if defined(HAVE_LIBCUDF)
cudf::table output_to_cudf(benchmark_device_buffers& device_buffers);
#endif // defined(HAVE_LIBCUDF)
void print_results();
void to_csv(ParserOutputHost<BaseAction>& output_hosts);
#if defined(HAVE_LIBCUDF)
void to_csv_libcudf(std::string& filename, cudf::table& cudf_table);
#endif // defined(HAVE_LIBCUDF)


class OutputIndicesIterator
{
public:

    // Required iterator traits
    typedef OutputIndicesIterator                        self_type;              ///< My own type
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
		int inner_offset = 1;
		//undefined behavior for 2 byte jsons. e.g. \n[]\n or \n{}\n
		//Windows endline format with \n\r is not supported and will result in udefined behavior
		uint32_t mask = __vcmpeq4(val.value, '\n\n\n\n') | __vcmpeq4(val.value, '\r\r\r\r');
		switch (mask)
		{
		case 0xFF'00'00'00u:
			inner_offset = 4;
			break;
		case 0x00'FF'00'00u:
			inner_offset = 3;
			break;
		case 0x00'00'FF'00u:
			inner_offset = 2;
			break;
		case 0x00'00'00'FFu:
			inner_offset = 1;
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

struct IsNewLine
{
	__device__ __forceinline__ bool operator()(const cub::KeyValuePair<ptrdiff_t, uint32_t> c) const {
		return __vcmpeq4(c.value, '\n\n\n\n') | __vcmpeq4(c.value, '\r\r\r\r');
	}
};

template <int GroupSizeT>
benchmark_device_buffers initialize_buffers(benchmark_input& input)
{
  	using GroupSize = mp_int<GroupSizeT>;
  	constexpr int GROUP_SIZE = GroupSizeT;
  	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
  	using GroupCount = mp_int<GROUP_COUNT>;
  	using MC = EmptyMemoryConfiguration;
  	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
  	using PC = ParserConfiguration<RT, MC>;
  	using PK = ParserKernel<PC, BaseAction>;
  	using M3 = typename PK::M3;
  	using BUF = typename M3::ReadOnlyBuffer;
	using KC = typename PK::KC;
	using OM = typename KC::OM;

	cudaEventRecord(gpu_memory_checkpoint, stream);
	benchmark_device_buffers result;
	result.count = input.count;
	result.parser_output_buffers = ParserOutputDevice<BaseAction>(nullptr, result.count);
	cudaMalloc(&result.readonly_buffers, sizeof(BUF));
	cudaMalloc(&result.input_buffer, input.data.size());
	cudaMalloc(&result.indices_buffer, sizeof(InputIndex) * (input.count + 1));
	cudaMalloc(&result.err_buffer, sizeof(ParsingError) * input.count);
	cudaMalloc(&result.output_buffers, sizeof(void*) * 20);

	vector<void*> output_buffers(20);
	for (int i = 0; i < 20; ++i)
	{
		output_buffers[i] = result.parser_output_buffers.m_d_outputs[i].data().get();
	}

	cudaMemcpyAsync(result.input_buffer, input.data.data(), input.data.size(), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(result.output_buffers, output_buffers.data(), sizeof(void*) * 20, cudaMemcpyHostToDevice, stream);

	find_newlines(result.input_buffer, input.data.size(), result.indices_buffer, input.count);

	return result;
}

template <int GroupSizeT>
void launch_kernel(benchmark_device_buffers& device_buffers)
{
  	using GroupSize = mp_int<GroupSizeT>;
  	constexpr int GROUP_SIZE = GroupSizeT;
  	constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
  	using GroupCount = mp_int<GROUP_COUNT>;

  	using MC = EmptyMemoryConfiguration;
  	using RT = RuntimeConfiguration<GroupSize, GroupCount>;
  	using PC = ParserConfiguration<RT, MC>;
  	using PK = ParserKernel<PC, BaseAction>;

	cudaEventRecord(gpu_parsing_checkpoint, stream);
	PK pk(nullptr, stream);

	pk.Run(
		device_buffers.input_buffer,
		device_buffers.indices_buffer,
		device_buffers.err_buffer,
		device_buffers.output_buffers,
		device_buffers.count,
		nullptr
	);
}

void main_metajson()
{
	init_gpu();
	benchmark_input input = get_input();
	cudaEventRecord(gpu_start, stream);
	benchmark_device_buffers device_buffers = initialize_buffers_dynamic(input);
	launch_kernel_dynamic(device_buffers, input.wg_size);
	auto host_output = copy_output(device_buffers);
#if defined(HAVE_LIBCUDF)
	auto cudf_table  = output_to_cudf(device_buffers);
#endif // defined(HAVE_LIBCUDF)
	cudaEventRecord(gpu_stop, stream);
	cudaEventSynchronize(gpu_stop);
	cpu_stop = chrono::high_resolution_clock::now();
	print_results();
	if (!g_args.output_csv.empty()) {
		to_csv(host_output);
#if defined(HAVE_LIBCUDF)
		to_csv_libcudf(g_args.output_csv, cudf_table);
#endif // defined(HAVE_LIBCUDF)
	}
}

#if defined(HAVE_LIBCUDF)
void init_libcudf();
cudf::io::json_reader_options prepare_libcudf(benchmark_input& input);
cudf::io::table_with_metadata parse_json_libcudf(cudf::io::json_reader_options const& json_in_opts);
//ParserOutputHost<BaseAction> copy_output_libcudf(cudf::io::table_with_metadata const& table_with_metadata);
void print_results_libcudf();

void main_libcudf()
{
    init_libcudf();
    benchmark_input input = get_input();
    cudaEventRecord(gpu_start, stream);
    auto json_reader_options = prepare_libcudf(input);
    auto libcudf_result = parse_json_libcudf(json_reader_options);
    //TODO: auto host_output = copy_output_libcudf(libcudf_result);
    cudaEventRecord(gpu_stop, stream);
    cudaEventSynchronize(gpu_stop);
    cpu_stop = chrono::high_resolution_clock::now();
    print_results_libcudf();
    // TODO: to csv for libcudf
    //if (!g_args.output_csv.empty())
    //    to_csv_libcudf(libcudf_result);
}
#endif /* HAVE_LIBCUDF */


int main(int argc, char** argv)
{
    cout << "INITIALIZATION\n";
    parse_args(argc, argv);

    cout << "META-JSON-PARSER\n";
    main_metajson();

#if defined(HAVE_LIBCUDF)
    // NOTE: must be second
    cout << "\nLIBCUDF (skipped)\n";
    //main_libcudf();
#else
    cout << "LIBCUDF not available, or not configured\n";
#endif /* HAVE_LIBCUDF */
}

void usage()
{
	cout << "usage: benchmark JSONLINES_FILE JSON_COUNT\n"
		 << "       [--ws=32|16|8] [-o=CSV_FILENAME] [-b]\n";
	exit(1);
}

void parse_args(int argc, char** argv)
{
	int pos_arg = 1;
	int wg_size = 32;
	g_args.error_check = false;
	try
	{
		for (int i = 1; i < argc; ++i)
		{
			std::string opt(argv[i]);
			if (opt[0] == '-')
			{
				if (opt.rfind("--ws=", 0) == 0)
					wg_size = std::stoi(opt.substr(5));
				else if (opt.rfind("-o=", 0) == 0)
					g_args.output_csv = opt.substr(3);
				else if (opt == "-b")
					g_args.error_check = true;
				else
				{
					cout << "Unknown option.\n";
					usage();
				}
			}
			else
			{
				switch (pos_arg)
				{
				case 1:
					g_args.filename = opt;
					break;
				case 2:
					g_args.count = std::stoi(opt);
					break;
				default:
					cout << "Exactly 2 positional arguments are supported.";
					usage();
					break;
				}
				++pos_arg;
			}
		}
	}
	catch (...)
	{
		cout << "Error in parsing arguments\n";
		usage();
	}
	if (pos_arg < 3)
	{
		cout << "Not enough positional arguments.\n";
		usage();
	}
	switch (wg_size)
	{
	case 32:
		g_args.wg_size = workgroup_size::W32;
		break;
	case 16:
		g_args.wg_size = workgroup_size::W16;
		break;
	case 8:
		g_args.wg_size = workgroup_size::W8;
		break;
	default:
		cout << "Only allowed workgroup sizes are: 32, 16 and 8.\n";
		usage();
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
	cpu_start = chrono::high_resolution_clock::now();
	file.read(data.data(), data.size());
	return benchmark_input
	{
		std::move(data),
		g_args.count,
		g_args.wg_size
	};
}

void init_gpu()
{
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_memory_checkpoint);
	cudaEventCreate(&gpu_preprocessing_checkpoint);
	cudaEventCreate(&gpu_parsing_checkpoint);
	cudaEventCreate(&gpu_output_checkpoint);
	cudaEventCreate(&gpu_error_checkpoint);
#if defined(HAVE_LIBCUDF)
	cudaEventCreate(&gpu_convert_checkpoint);
#endif // defined(HAVE_LIBCUDF)
	cudaEventCreate(&gpu_stop);
	cudaStreamCreate(&stream);
}

#if defined(HAVE_LIBCUDF)
void init_libcudf()
{
    /*
     * events and streams are created by init_gpu()
     * events needed:
     * - gpu_start
     * - gpu_parsing_checkpoint
     * - gpu_stop
     */
}
#endif /* HAVE_LIBCUDF */

void find_newlines(char* d_input, size_t input_size, InputIndex* d_indices, int count)
{
	cudaEventRecord(gpu_preprocessing_checkpoint, stream);
	InputIndex just_zero = 0;
	cudaMemcpyAsync(d_indices, &just_zero, sizeof(InputIndex), cudaMemcpyHostToDevice, stream);
	
	cub::ArgIndexInputIterator<uint32_t*> arg_iter(reinterpret_cast<uint32_t*>(d_input));
	OutputIndicesIterator out_iter(d_indices + 1); // +1, we need to add 0 at index 0
	
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
		IsNewLine(),
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
		IsNewLine(),
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

struct NoError
{
	__device__ __host__ bool operator()(ParsingError e)
	{
		return ParsingError::None == e;
	}
};

ParserOutputHost<BaseAction> copy_output(benchmark_device_buffers& device_buffers)
{
	cudaEventRecord(gpu_output_checkpoint, stream);
	vector<ParsingError> temp_err(device_buffers.count);
	cudaMemcpyAsync(temp_err.data(), device_buffers.err_buffer, sizeof(ParsingError) * device_buffers.count, cudaMemcpyDeviceToHost, stream);
	ParserOutputHost<BaseAction> output = device_buffers.parser_output_buffers.CopyToHost(stream);

	if (g_args.error_check)
	{
		cudaEventRecord(gpu_error_checkpoint, stream);
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

#if defined(HAVE_LIBCUDF)
/**
 * This function converts benchmark results on GPU to cuDF format.
 * @param device_buffers represents benchmark results on GPU
 * @return cudf::table
 */
cudf::table output_to_cudf(benchmark_device_buffers& device_buffers) {
    cudaEventRecord(gpu_convert_checkpoint, stream);
    return device_buffers.parser_output_buffers.ToCudf(stream);
}
#endif // defined(HAVE_LIBCUDF)

benchmark_device_buffers initialize_buffers_dynamic(benchmark_input& input)
{
	switch (input.wg_size)
	{
	default:
	case workgroup_size::W32:
		return initialize_buffers<32>(input);
	case workgroup_size::W16:
		return initialize_buffers<16>(input);
	case workgroup_size::W8:
		return initialize_buffers<8>(input);
	}
}

#if defined(HAVE_LIBCUDF)
cudf::io::json_reader_options prepare_libcudf(benchmark_input& input)
{
    cudaEventRecord(gpu_memory_checkpoint, stream);

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
#endif /* HAVE_LIBCUDF */

void to_csv(ParserOutputHost<BaseAction>& output_hosts)
{
	if (g_args.output_csv.empty())
		return;
	cout << "Saving results to " << g_args.output_csv << "\n";
	OutputPrinter<BaseAction, PrinterMap> printer;
	std::ofstream csv(g_args.output_csv);
	printer.ToCsv(csv, output_hosts);
}

#if defined(HAVE_LIBCUDF)
void to_csv_libcudf(std::string& filename, cudf::table& cudf_table)
{
	if (filename.empty())
		return;

	cout << "Saving results to " << filename << " (via libcudf)";
	cout.flush();

	cudf::io::csv_writer_options csv_out_opts =
		cudf::io::csv_writer_options::builder(cudf::io::sink_info{filename},
											  cudf_table.view())
			.inter_column_delimiter(',')
			.include_header(false);
	cudf::io::write_csv(csv_out_opts);

	cout << "\n";
}
#endif /* HAVE_LIBCUDF */

void launch_kernel_dynamic(benchmark_device_buffers& device_buffers, workgroup_size wg_size)
{
	switch (wg_size)
	{
	case workgroup_size::W32:
		return launch_kernel<32>(device_buffers);
	case workgroup_size::W16:
		return launch_kernel<16>(device_buffers);
	case workgroup_size::W8:
		return launch_kernel<8>(device_buffers);
	default:
		break;
	}
}

#if defined(HAVE_LIBCUDF)
cudf::io::table_with_metadata parse_json_libcudf(cudf::io::json_reader_options const& json_in_opts)
{
    cudaEventRecord(gpu_parsing_checkpoint, stream);

    auto result = cudf::io::read_json(json_in_opts);
    return result;
}
#endif /* HAVE_LIBCUDF */

void print_results()
{
	float ms;
	int64_t cpu_ns = (cpu_stop - cpu_start).count();
	cudaEventElapsedTime(&ms, gpu_start, gpu_stop);
	int64_t gpu_total = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_start, gpu_memory_checkpoint);
	int64_t gpu_init = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_memory_checkpoint, gpu_preprocessing_checkpoint);
	int64_t gpu_memory = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_preprocessing_checkpoint, gpu_parsing_checkpoint);
	int64_t gpu_preproc = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_parsing_checkpoint, gpu_output_checkpoint);
	int64_t gpu_parsing = static_cast<int64_t>(ms * 1'000'000.0);
	if (!g_args.error_check)
#if defined(HAVE_LIBCUDF)
        cudaEventElapsedTime(&ms, gpu_output_checkpoint, gpu_convert_checkpoint);
#else  // !defined(HAVE_LIBCUDF)
		cudaEventElapsedTime(&ms, gpu_output_checkpoint, gpu_stop);
#endif // !defined(HAVE_LIBCUDF)
	else
		cudaEventElapsedTime(&ms, gpu_output_checkpoint, gpu_error_checkpoint);
	int64_t gpu_output = static_cast<int64_t>(ms * 1'000'000.0);
	if (g_args.error_check)
#if defined(HAVE_LIBCUDF)
        cudaEventElapsedTime(&ms, gpu_error_checkpoint, gpu_convert_checkpoint);
#else  // !defined(HAVE_LIBCUDF)
		cudaEventElapsedTime(&ms, gpu_error_checkpoint, gpu_stop);
#endif // !defined(HAVE_LIBCUDF)
	int64_t gpu_error = static_cast<int64_t>(ms * 1'000'000.0);
#if defined(HAVE_LIBCUDF)
	cudaEventElapsedTime(&ms, gpu_convert_checkpoint, gpu_stop);
	int64_t gpu_convert = static_cast<int64_t>(ms * 1'000'000.0);
#endif // defined(HAVE_LIBCUDF)

	const int c1 = 40;
	const int c2 = 10;

	cout
		<< "Time measured by GPU:\n"
		<< setw(c1) << left  << "+ Initialization: "
		<< setw(c2) << right << gpu_init << " ns\n"
		<< setw(c1) << left  << "+ Memory allocation and copying: "
		<< setw(c2) << right << gpu_memory << " ns\n"
		<< setw(c1) << left  << "+ Finding newlines offsets (indices): "
		<< setw(c2) << right << gpu_preproc << " ns\n"
		<< setw(c1) << left  << "+ Parsing json: "
		<< setw(c2) << right << gpu_parsing << " ns\n"
		<< setw(c1) << left  << "+ Copying output: "
		<< setw(c2) << right << gpu_output << " ns\n"
		;
	if (g_args.error_check)
	cout
		<< setw(c1) << left  << "+ Checking parsing errors: "
		<< setw(c2) << right << gpu_error << " ns\n";
#if defined(HAVE_LIBCUDF)
	cout
	    << setw(c1) << left  << "+ Converting to cuDF format: "
        << setw(c2) << right << gpu_convert << " ns\n";
#endif // defined(HAVE_LIBCUDF)
	cout
		<< setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
		<< setw(c1) << left  << "Total time measured by GPU: "
		<< setw(c2) << right << gpu_total << " ns\n"
		<< setw(c1) << left  << "Total time measured by CPU: "
		<< setw(c2) << right << cpu_ns << " ns\n"
		;
}

#if defined(HAVE_LIBCUDF)
void print_results_libcudf()
{
    float ms;
    int64_t cpu_ns = (cpu_stop - cpu_start).count();
    cudaEventElapsedTime(&ms, gpu_start, gpu_stop);
    int64_t gpu_total = static_cast<int64_t>(ms * 1'000'000.0);
    cudaEventElapsedTime(&ms, gpu_start, gpu_memory_checkpoint);
    int64_t gpu_init = static_cast<int64_t>(ms * 1'000'000.0);
    cudaEventElapsedTime(&ms, gpu_memory_checkpoint, gpu_parsing_checkpoint);
    int64_t gpu_prep = static_cast<int64_t>(ms * 1'000'000.0);
    cudaEventElapsedTime(&ms, gpu_parsing_checkpoint, gpu_stop);
    int64_t gpu_parsing = static_cast<int64_t>(ms * 1'000'000.0);

    const int c1 = 40; // description width
    const int c2 = 10; // results width

    cout
            << "Time measured by GPU:\n"
            << setw(c1) << left  << "+ Initialization: "
            << setw(c2) << right << gpu_init << " ns\n"
            << setw(c1) << left  << "+ Building input options: "
            << setw(c2) << right << gpu_prep << " ns\n"
            << setw(c1) << left  << "+ Parsing json: "
            << setw(c2) << right << gpu_parsing << " ns\n"
            //<< setw(c1) << left  << "+ Copying output: "
            //<< setw(c2) << right << gpu_output << " ns\n"
            ;

    cout
        << setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
        << setw(c1) << left  << "Total time measured by GPU: "
        << setw(c2) << right << gpu_total << " ns\n"
        << setw(c1) << left  << "Total time measured by CPU: "
        << setw(c2) << right << cpu_ns << " ns\n"
        ;
}
#endif /* HAVE_LIBCUDF */