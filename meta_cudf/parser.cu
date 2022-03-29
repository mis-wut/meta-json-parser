//#include "opt1/meta_def.cuh"
#include <fstream>
#include <memory>

#include <boost/mp11.hpp>
#include <cudf/io/types.hpp>
#include <thrust/logical.h>
#include <iomanip>
#include <meta_json_parser/parser_output_device.cuh>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/action/jstring.cuh>

#include <meta_def.cuh>

using namespace std;
using namespace boost::mp11;

cudaStream_t stream;

enum class end_of_line {
    unknown,
    uniks, //< LF, or "\n": end-of-line convention used by Unix
    win   //< CRLF, or "\r\n": end-of-line convention used by MS Windows
};

namespace EndOfLine
{
    struct Unix {};
    struct Win {};
}

struct NoError
{
    __device__ __host__ bool operator()(ParsingError e)
    {
        return ParsingError::None == e;
    }
};

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
struct IsNewLine
{
    __device__ __forceinline__ bool operator()(const cub::KeyValuePair<ptrdiff_t, uint32_t> c) const {
        return LineEndingHelper<EndOfLineT>::is_newline(c.value);
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

struct benchmark_input
{
    vector<char> data;
    int count;
    end_of_line eol;
    int bytes_per_string;
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

    vector<void*> host_output_buffers;
};

benchmark_input get_input(const char* filename, int input_count);
KernelLaunchConfiguration prepare_dynamic_config(benchmark_input& input);
benchmark_device_buffers initialize_buffers(benchmark_input& input, KernelLaunchConfiguration* conf);
end_of_line detect_eol(benchmark_input& input);
void launch_kernel(benchmark_device_buffers& device_buffers);
ParserOutputHost<BaseAction> copy_output(benchmark_device_buffers& device_buffers);

template<class EndOfLineT>
void find_newlines(char* d_input, size_t input_size, InputIndex* d_indices, int count)
{
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
        throw runtime_error("Invalid number of new lines.");
    }

    cudaFree(d_temp_storage);
    cudaFree(d_num_selected);
}

cudf::io::table_with_metadata generate_example_metadata(const char* filename, int count) {
	cudaStreamCreate(&stream);

    auto input = get_input(filename, count);

    KernelLaunchConfiguration conf = prepare_dynamic_config(input);
    benchmark_device_buffers device_buffers = initialize_buffers(input, &conf);
    launch_kernel(device_buffers);
    auto host_output = copy_output(device_buffers);
    auto cudf_table  = device_buffers.parser_output_buffers.ToCudf(stream);

    vector<string> column_names(cudf_table.num_columns());

    generate(column_names.begin(), column_names.end(), [i = 1]() mutable {
        return "Column " + to_string(i++);
    });

    cudf::io::table_metadata metadata{column_names};

    return cudf::io::table_with_metadata{
        make_unique<cudf::table>(cudf_table),
        metadata
    };
}

ParserOutputHost<BaseAction> copy_output(benchmark_device_buffers& device_buffers)
{
    vector<ParsingError> temp_err(device_buffers.count);
    cudaMemcpyAsync(temp_err.data(), device_buffers.err_buffer, sizeof(ParsingError) * device_buffers.count, cudaMemcpyDeviceToHost, stream);
    ParserOutputHost<BaseAction> output = device_buffers.parser_output_buffers.CopyToHost(stream);

    //if (false)
    //{
    //    bool correct = thrust::all_of(
    //        thrust::cuda::par.on(stream),
    //        device_buffers.err_buffer,
    //        device_buffers.err_buffer + device_buffers.count,
    //        NoError()
    //    );
    //    if (!correct)
    //    {
    //        cerr << "Parsing errors!\n";
    //    }
    //}

    return output;
}

void launch_kernel(benchmark_device_buffers& device_buffers)
{
    using GroupSize = WorkGroupSize;
    constexpr int GROUP_SIZE = WorkGroupSize::value;
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
        device_buffers.host_output_buffers.data()
    );
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

KernelLaunchConfiguration prepare_dynamic_config(benchmark_input& input)
{
    KernelLaunchConfiguration conf;

    using DynamicStringActions = mp_copy_if_q<
        ActionIterator<BaseAction>,
        mp_bind<
            mp_similar,
            JStringDynamicCopy<void>,
            _1
        >
    >;

    using DynamicStringActionsV2 = mp_copy_if_q<
        ActionIterator<BaseAction>,
        mp_bind<
            mp_similar,
            JStringDynamicCopyV2<void>,
            _1
        >
    >;

    using DynamicStringActionsV3 = mp_copy_if_q<
        ActionIterator<BaseAction>,
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
        conf.SetDynamicSize<BaseAction, Tag>(input.bytes_per_string);
    });

    mp_for_each<DynamicStringActionsV3>([&conf, &input](auto a) {
        using Action = decltype(a);
        using TagInternal = typename Action::DynamicStringInternalRequestTag;
        conf.SetDynamicSize<BaseAction, TagInternal>(input.bytes_per_string);
        using Tag = typename Action::DynamicStringRequestTag;
        conf.SetDynamicSize<BaseAction, Tag>(input.bytes_per_string);
    });

    return std::move(conf);
}

benchmark_device_buffers initialize_buffers(benchmark_input& input, KernelLaunchConfiguration* conf)
{
    using GroupSize = WorkGroupSize;
    constexpr int GROUP_SIZE = WorkGroupSize::value;
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

    benchmark_device_buffers result;
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
            throw std::runtime_error("Unknown end of line character");
    }

    return result;
}

benchmark_input get_input(const char* filename, int input_count)
{
    ifstream file(filename, ifstream::ate | ifstream::binary);
    if (!file.good())
    {
        cout << "Error reading file \"" << filename << "\".\n";
        throw std::runtime_error("Error reading file.");
    }
    vector<char> data(file.tellg());
    file.seekg(0);
    file.read(data.data(), static_cast<streamsize>(data.size()));

    return benchmark_input
        {
            std::move(data),
            input_count,
            end_of_line::unknown,
            32
        };
}


