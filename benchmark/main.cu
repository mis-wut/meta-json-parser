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

#define STR_FUN_STA(KEY, SIZE) JStringStaticCopy<mp_int<SIZE>, KEY>
#define STR_FUN_DYN(KEY, ...) JStringDynamicCopy<KEY>

#define GET_ACTION(STR_FUN)\
JDict < mp_list <\
	mp_list<K_L1_date, STR_FUN(K_L1_date, 32)>,\
	mp_list<K_L1_lat, JNumber<uint32_t, K_L1_lat>>,\
	mp_list<K_L1_lon, JNumber<uint32_t, K_L1_lon>>,\
	mp_list<K_L1_is_checked, JBool<uint8_t, K_L1_is_checked>>,\
	mp_list<K_L1_name, STR_FUN(K_L1_name, 32)>,\
	mp_list<K_L1_1_date, STR_FUN(K_L1_1_date, 32)>,\
	mp_list<K_L1_1_lat, JNumber<uint32_t, K_L1_1_lat>>,\
	mp_list<K_L1_1_lon, JNumber<uint32_t, K_L1_1_lon>>,\
	mp_list<K_L1_1_is_checked, JBool<uint8_t, K_L1_1_is_checked>>,\
	mp_list<K_L1_1_name, STR_FUN(K_L1_1_name, 32)>,\
	mp_list<K_L1_2_date, STR_FUN(K_L1_2_date, 32)>,\
	mp_list<K_L1_2_lat, JNumber<uint32_t, K_L1_2_lat>>,\
	mp_list<K_L1_2_lon, JNumber<uint32_t, K_L1_2_lon>>,\
	mp_list<K_L1_2_is_checked, JBool<uint8_t, K_L1_2_is_checked>>,\
	mp_list<K_L1_2_name, STR_FUN(K_L1_2_name, 32)>,\
	mp_list<K_L1_3_date, STR_FUN(K_L1_3_date, 32)>,\
	mp_list<K_L1_3_lat, JNumber<uint32_t, K_L1_3_lat>>,\
	mp_list<K_L1_3_lon, JNumber<uint32_t, K_L1_3_lon>>,\
	mp_list<K_L1_3_is_checked, JBool<uint8_t, K_L1_3_is_checked>>,\
	mp_list<K_L1_3_name, STR_FUN(K_L1_3_name, 32)>\
>>

using BaseActionStatic = GET_ACTION(STR_FUN_STA);
using BaseActionDynamic = GET_ACTION(STR_FUN_DYN);

enum workgroup_size { W32, W16, W8 };
enum end_of_line { unknown, unix, win };

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
	int count;
	workgroup_size wg_size;
	std::string output_csv;
	bool error_check;
	int bytes_per_string;
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
cudaEvent_t gpu_stop;
cudaStream_t stream;

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
	cudaEventRecord(gpu_preprocessing_checkpoint, stream);
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

	cudaEventRecord(gpu_memory_checkpoint, stream);
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
	case end_of_line::unix:
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
	default:
		break;
	}
}

template<class BaseActionT>
KernelLaunchConfiguration prepare_dynamic_config(benchmark_input& input)
{
	KernelLaunchConfiguration conf;

	using DynamicStringActions = boost::mp11::mp_copy_if_q<
		ActionIterator<BaseActionT>,
		boost::mp11::mp_bind<
			boost::mp11::mp_similar,
			JStringDynamicCopy<void>,
			boost::mp11::_1
		>
	>;

	boost::mp11::mp_for_each<DynamicStringActions>([&conf, &input](auto a) {
		using Action = decltype(a);
		using Tag = typename Action::DynamicStringRequestTag;
		conf.SetDynamicSize<BaseActionT, Tag>(input.bytes_per_string);
	});

	return std::move(conf);
}

template<class BaseActionT>
ParserOutputHost<BaseActionT> copy_output(benchmark_device_buffers<BaseActionT>& device_buffers)
{
	using BaseAction = BaseActionT;
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

template<class BaseActionT>
void to_csv(ParserOutputHost<BaseActionT>& output_hosts)
{
	if (g_args.output_csv.empty())
		return;
	cout << "Saving results to " << g_args.output_csv << ".";
	output_hosts.DropToCsv(g_args.output_csv.c_str());;
}

template<class BaseActionT>
void main_templated(benchmark_input& input)
{
	cudaEventRecord(gpu_start, stream);
	KernelLaunchConfiguration conf = prepare_dynamic_config<BaseActionT>(input);
	benchmark_device_buffers<BaseActionT> device_buffers = initialize_buffers_dynamic<BaseActionT>(input, &conf);
	launch_kernel_dynamic<BaseActionT>(device_buffers, input.wg_size);
	auto host_output = copy_output<BaseActionT>(device_buffers);
	cudaEventRecord(gpu_stop, stream);
	cudaEventSynchronize(gpu_stop);
	cpu_stop = chrono::high_resolution_clock::now();
	print_results();
	if (!g_args.output_csv.empty())
		to_csv<BaseActionT>(host_output);
}

int main(int argc, char** argv)
{
	init_gpu();
	parse_args(argc, argv);
	benchmark_input input = get_input();
	if (input.bytes_per_string == 0)
	{
		cout << "Using STATIC string copy.\n";
		main_templated<BaseActionStatic>(input);
	}
	else
	{
		cout << "Using DYNAMIC string copy.\n";
		main_templated<BaseActionDynamic>(input);
	}
	return 0;
}

void usage()
{
	cout << "usage: benchmark JSONLINES_FILE JSON_COUNT\n"
         << "  --ws=32|16|8      Workgroup size. Default = 32\n"
         << "  -o=CSV_FILENAME   Name for an parsed output CSV file.\n"
         << "                    If ommited no output is saved.\n"
         << "  -b                Enable error check. If there was a\n"
         << "                    parsing error, message will be printed.\n"
         << "  -s=BYTES          Bytes allocated per dynamic string.\n"
         << "                    String that are too long will be truncated.\n"
         << "                    If not provided, then strings with static\n"
         << "                    length will be used.\n"
		;
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
				else if (opt.rfind("-s=", 0) == 0)
					g_args.bytes_per_string = std::stoi(opt.substr(3));
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
	cudaEventCreate(&gpu_stop);
	cudaStreamCreate(&stream);
}

end_of_line detect_eol(benchmark_input& input)
{
	auto found = std::find_if(input.data.begin(), input.data.end(), [](char& c) {
		return c == '\r' || c == '\n';
	});
	if (found == input.data.end())
		return end_of_line::unknown;
	if (*found == '\n')
		return end_of_line::unix;
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
	cudaEventElapsedTime(&ms, gpu_start, gpu_memory_checkpoint);
	int64_t gpu_init = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_memory_checkpoint, gpu_preprocessing_checkpoint);
	int64_t gpu_memory = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_preprocessing_checkpoint, gpu_parsing_checkpoint);
	int64_t gpu_preproc = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_parsing_checkpoint, gpu_post_hooks_checkpoint);
	int64_t gpu_parsing = static_cast<int64_t>(ms * 1'000'000.0);
	cudaEventElapsedTime(&ms, gpu_post_hooks_checkpoint, gpu_output_checkpoint);
	int64_t gpu_post_hooks = static_cast<int64_t>(ms * 1'000'000.0);
	if (!g_args.error_check)
		cudaEventElapsedTime(&ms, gpu_output_checkpoint, gpu_stop);
	else
		cudaEventElapsedTime(&ms, gpu_output_checkpoint, gpu_error_checkpoint);
	int64_t gpu_output = static_cast<int64_t>(ms * 1'000'000.0);
	if (g_args.error_check)
		cudaEventElapsedTime(&ms, gpu_error_checkpoint, gpu_stop);
	int64_t gpu_error = static_cast<int64_t>(ms * 1'000'000.0);

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
		<< setw(c1) << left  << "+ Parsing total (sum of the following): "
		<< setw(c2) << right << gpu_parsing + gpu_post_hooks << " ns\n"
		<< setw(c1) << left  << "  - JSON processing: "
		<< setw(c2) << right << gpu_parsing << " ns\n"
		<< setw(c1) << left  << "  - Post kernel hooks: "
		<< setw(c2) << right << gpu_post_hooks << " ns\n"
		<< setw(c1) << left  << "+ Copying output: "
		<< setw(c2) << right << gpu_output << " ns\n"
		;
	if (g_args.error_check)
	cout
		<< setw(c1) << left  << "+ Checking parsing errors: "
		<< setw(c2) << right << gpu_error << " ns\n";
	cout
		<< setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
		<< setw(c1) << left  << "Total time measured by GPU: "
		<< setw(c2) << right << gpu_total << " ns\n"
		<< setw(c1) << left  << "Total time measured by CPU: "
		<< setw(c2) << right << cpu_ns << " ns\n"
		;
}

