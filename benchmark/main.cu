#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <iomanip>
#include <chrono>
#include <functional>
#include <cuda_runtime_api.h>
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
	array<char*, 8> output_char_buffers;
	array<uint32_t*, 8> output_uint32_buffers;
	array<uint8_t*, 4> output_uint8_buffers;
	char* readonly_buffers;
	char* input_buffer;
	InputIndex* indices_buffer;
	ParsingError* err_buffer;
	void** output_buffers;
	int count;
};

chrono::high_resolution_clock::time_point cpu_start;
chrono::high_resolution_clock::time_point cpu_stop;
cudaEvent_t gpu_start;
cudaEvent_t gpu_memory_checkpoint;
cudaEvent_t gpu_preprocessing_checkpoint;
cudaEvent_t gpu_parsing_checkpoint;
cudaEvent_t gpu_output_checkpoint;
cudaEvent_t gpu_stop;
cudaStream_t stream;

void usage();
void init_gpu();
benchmark_input get_input(int argc, char** argv);
benchmark_device_buffers initialize_buffers_dynamic(benchmark_input& input);
void launch_kernel_dynamic(benchmark_device_buffers& device_buffers, workgroup_size wg_size);
void find_newlines(char* d_input, size_t input_size, InputIndex* d_indices, int count);
void copy_output(benchmark_device_buffers& device_buffers);
void print_results();

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
	for (auto& buf : result.output_char_buffers)
	{
		cudaMalloc(&buf, CHARS_PER_JSTRING * input.count);
	}
	for (auto& buf : result.output_uint32_buffers)
	{
		cudaMalloc(&buf, sizeof(uint32_t) * input.count);
	}
	for (auto& buf : result.output_uint8_buffers)
	{
		cudaMalloc(&buf, sizeof(uint8_t) * input.count);
	}
	cudaMalloc(&result.readonly_buffers, sizeof(BUF));
	cudaMalloc(&result.input_buffer, input.data.size());
	cudaMalloc(&result.indices_buffer, sizeof(InputIndex) * (input.count + 1));
	cudaMalloc(&result.err_buffer, sizeof(ParsingError) * input.count);
	cudaMalloc(&result.output_buffers, sizeof(void*) * 20);

	vector<void*> output_buffers(20);
	output_buffers[OM::template TagIndex<K_L1_date>::value  ] = result.output_char_buffers[0];
	output_buffers[OM::template TagIndex<K_L1_name>::value  ] = result.output_char_buffers[1];
	output_buffers[OM::template TagIndex<K_L1_1_date>::value] = result.output_char_buffers[2];
	output_buffers[OM::template TagIndex<K_L1_1_name>::value] = result.output_char_buffers[3];
	output_buffers[OM::template TagIndex<K_L1_2_date>::value] = result.output_char_buffers[4];
	output_buffers[OM::template TagIndex<K_L1_2_name>::value] = result.output_char_buffers[5];
	output_buffers[OM::template TagIndex<K_L1_3_date>::value] = result.output_char_buffers[6];
	output_buffers[OM::template TagIndex<K_L1_3_name>::value] = result.output_char_buffers[7];
	output_buffers[OM::template TagIndex<K_L1_lat>::value  ] = result.output_uint32_buffers[0];
	output_buffers[OM::template TagIndex<K_L1_lon>::value  ] = result.output_uint32_buffers[1];
	output_buffers[OM::template TagIndex<K_L1_1_lat>::value] = result.output_uint32_buffers[2];
	output_buffers[OM::template TagIndex<K_L1_1_lon>::value] = result.output_uint32_buffers[3];
	output_buffers[OM::template TagIndex<K_L1_2_lat>::value] = result.output_uint32_buffers[4];
	output_buffers[OM::template TagIndex<K_L1_2_lon>::value] = result.output_uint32_buffers[5];
	output_buffers[OM::template TagIndex<K_L1_3_lat>::value] = result.output_uint32_buffers[6];
	output_buffers[OM::template TagIndex<K_L1_3_lon>::value] = result.output_uint32_buffers[7];
	output_buffers[OM::template TagIndex<K_L1_is_checked>::value  ] = result.output_uint8_buffers[0];
	output_buffers[OM::template TagIndex<K_L1_1_is_checked>::value] = result.output_uint8_buffers[1];
	output_buffers[OM::template TagIndex<K_L1_2_is_checked>::value] = result.output_uint8_buffers[2];
	output_buffers[OM::template TagIndex<K_L1_3_is_checked>::value] = result.output_uint8_buffers[3];

	BUF readonly_buffer;
	M3::FillReadOnlyBuffer(readonly_buffer);

	cudaMemcpyAsync(result.readonly_buffers, &readonly_buffer, sizeof(BUF), cudaMemcpyHostToDevice, stream);
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
  	using M3 = typename PK::M3;
  	using BUF = typename M3::ReadOnlyBuffer;
	const unsigned int BLOCKS_COUNT = (device_buffers.count + GROUP_COUNT - 1) / GROUP_COUNT;

	cudaEventRecord(gpu_parsing_checkpoint, stream);
	typename PK::Launcher(&_parser_kernel<PC, BaseAction>)(BLOCKS_COUNT, stream)(
		reinterpret_cast<BUF*>(device_buffers.readonly_buffers),
		device_buffers.input_buffer,
		device_buffers.indices_buffer,
		device_buffers.err_buffer,
		device_buffers.output_buffers,
		device_buffers.count
	);
}

constexpr bool check_err_code = false;

int main(int argc, char** argv)
{
	init_gpu();
	benchmark_input input = get_input(argc, argv);
	cudaEventRecord(gpu_start, stream);
	benchmark_device_buffers device_buffers = initialize_buffers_dynamic(input);
	launch_kernel_dynamic(device_buffers, input.wg_size);
	copy_output(device_buffers);
	cudaEventRecord(gpu_stop, stream);
	cudaEventSynchronize(gpu_stop);
	cpu_stop = chrono::high_resolution_clock::now();
	print_results();
}

void usage()
{
	cout << "usage: benchmark JSONLINES_FILE JSON_COUNT [WORKGROUP_SIZE=32]\n";
	exit(1);
}

benchmark_input get_input(int argc, char** argv)
{
	if (argc < 3 || argc > 4)
		usage();
	const char* filename_cstr = argv[1];
	const char* count_cstr = argv[2];
	const char* wg_cstr = argv[3];
	ifstream file(filename_cstr, std::ifstream::ate | std::ifstream::binary);
	if (!file.good())
	{
		cout << "Error reading file \"" << filename_cstr << "\".\n";
		usage();
	}
	vector<char> data(file.tellg());
	int count = 0;
	workgroup_size wg_size = workgroup_size::W32;
	try
	{
		count = std::stoi(count_cstr);
	}
	catch (...)
	{
		cout << "Error parsing count of Jsons \"" << count_cstr << "\".\n";
		usage();
	}
	if (argc >= 4)
	{
		try
		{
			auto wg = std::stoi(wg_cstr);
			switch (wg)
			{
			case 32:
				wg_size = workgroup_size::W32;
				break;
			case 16:
				wg_size = workgroup_size::W16;
				break;
			case 8:
				wg_size = workgroup_size::W8;
				break;
			default:
				cout << "Only allowed workgroup sizes are: 32, 16 and 8.\n";
				usage();
			}
		}
		catch (...)
		{
			cout << "Error parsing workgroup size \"" << wg_cstr << "\".\n";
			usage();
		}
	}
	file.seekg(0);
	//Start measuring CPU time with reading data form disk
	cpu_start = chrono::high_resolution_clock::now();
	file.read(data.data(), data.size());
	return benchmark_input
	{
		std::move(data),
		count,
		wg_size
	};
}

void init_gpu()
{
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_memory_checkpoint);
	cudaEventCreate(&gpu_preprocessing_checkpoint);
	cudaEventCreate(&gpu_parsing_checkpoint);
	cudaEventCreate(&gpu_output_checkpoint);
	cudaEventCreate(&gpu_stop);
	cudaStreamCreate(&stream);
}

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

void copy_output(benchmark_device_buffers& device_buffers)
{
	cudaEventRecord(gpu_output_checkpoint);
	vector<vector<char>> temp_char_bufs;
	vector<vector<uint32_t>> temp_uint32_bufs;
	vector<vector<uint8_t>> temp_uint8_bufs;
	vector<ParsingError> temp_err(device_buffers.count);
	for (auto& buf : device_buffers.output_char_buffers)
	{
		temp_char_bufs.push_back(vector<char>(CHARS_PER_JSTRING * device_buffers.count));
		cudaMemcpyAsync(temp_char_bufs.back().data(), buf, CHARS_PER_JSTRING * device_buffers.count, cudaMemcpyDeviceToHost, stream);
	}
	for (auto& buf : device_buffers.output_uint32_buffers)
	{
		temp_uint32_bufs.push_back(vector<uint32_t>(device_buffers.count));
		cudaMemcpyAsync(temp_uint32_bufs.back().data(), buf, sizeof(uint32_t) * device_buffers.count, cudaMemcpyDeviceToHost, stream);
	}
	for (auto& buf : device_buffers.output_uint8_buffers)
	{
		temp_uint8_bufs.push_back(vector<uint8_t>(device_buffers.count));
		cudaMemcpyAsync(temp_uint8_bufs.back().data(), buf, sizeof(uint8_t) * device_buffers.count, cudaMemcpyDeviceToHost, stream);
	}
	cudaMemcpyAsync(temp_err.data(), device_buffers.err_buffer, sizeof(ParsingError) * device_buffers.count, cudaMemcpyDeviceToHost, stream);

	if (check_err_code)
	{
		cudaStreamSynchronize(stream);
		if (!all_of(temp_err.begin(), temp_err.end(), [](ParsingError e) { return e == ParsingError::None; }))
		{
			cout << "Parsing errors!\n";
		}
	}
}

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
	cudaEventElapsedTime(&ms, gpu_output_checkpoint, gpu_stop);
	int64_t gpu_output = static_cast<int64_t>(ms * 1'000'000.0);

	const int c1 = 40;
	const int c2 = 10;

	cout
		<< "Time measured by GPU:\n"
		//<< setw(c1) << left << "Time measured by GPU:\n"
		<< setw(c1) << left << "+ Initialization: "
		<< setw(c2) << right << gpu_init << " ns\n"
		<< setw(c1) << left << "+ Memory allocation and copying: "
		<< setw(c2) << right << gpu_memory << " ns\n"
		<< setw(c1) << left << "+ Finding newlines offsets (indices): "
		<< setw(c2) << right << gpu_preproc << " ns\n"
		<< setw(c1) << left << "+ Parsing json: "
		<< setw(c2) << right << gpu_parsing << " ns\n"
		<< setw(c1) << left << "+ Copying output: "
		<< setw(c2) << right << gpu_output << " ns\n"
		<< setw(c1 + c2 + 4) << setfill('-') << "\n" << setfill(' ')
		<< setw(c1) << left << "Total time measured by GPU: "
		<< setw(c2) << right << gpu_total << " ns\n"
		<< setw(c1) << left << "Total time measured by CPU: "
		<< setw(c2) << right << cpu_ns << " ns\n"
		;
}

