#pragma once
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/kernel_launcher.cuh>
#include <meta_json_parser/kernel_launch_configuration.cuh>
#include <meta_json_parser/action_iterator.h>
#include <cstdint>
#include <type_traits>

template<class BaseActionT>
using CreateOutputConfig = OutputConfiguration<typename BaseActionT::OutputRequests>;

template<class ParserConfigurationT, class BaseActionT>
using CreateMemoryConfig = ExtendRequests<
	typename ParserConfigurationT::MemoryConfiguration,
	boost::mp11::mp_append<
		typename BaseActionT::MemoryRequests,
		typename CreateOutputConfig<BaseActionT>::MemoryRequests
	>
>;

template<class ParserConfigurationT, class BaseActionT>
using UpdateParserConfig = ParserConfiguration<
	typename ParserConfigurationT::RuntimeConfiguration,
	CreateMemoryConfig<ParserConfigurationT, BaseActionT>
>;

template<class ParserConfigurationT, class BaseActionT>
using CreateMetaMemoryManager = MetaMemoryManager<
	UpdateParserConfig<
		ParserConfigurationT,
		BaseActionT
	>
>;


template<class ActionT, typename = int>
struct HavePostKernelHook : std::false_type {};

template<class ActionT>
struct HavePostKernelHook<
	ActionT,
	decltype(
		ActionT::PostKernelHook(std::declval<int&>(), 0, nullptr),
		0
	)
> : std::true_type {};

template<class ParserConfigurationT, class BaseActionT>
__global__ void __launch_bounds__(1024, 2)
_parser_kernel(
	typename CreateMetaMemoryManager<ParserConfigurationT, BaseActionT>::ReadOnlyBuffer* readOnlyBuffer,
	const char* input,
	const InputIndex* indices,
	ParsingError* err,
	void** output,
	const uint32_t count);

template<class ParserConfigurationT, class BaseActionT>
struct ParserKernel
{
	using OC = CreateOutputConfig<BaseActionT>;
	using MC = CreateMemoryConfig<ParserConfigurationT, BaseActionT>;
	using PC = UpdateParserConfig<ParserConfigurationT, BaseActionT>;
	using M3 = CreateMetaMemoryManager<ParserConfigurationT, BaseActionT>;
  	using ROB = typename M3::ReadOnlyBuffer;
	using RT = typename PC::RuntimeConfiguration;
	using KC = KernelContext<PC, OC>;
	using Launcher = KernelLauncherFixedResources<
		typename RT::BlockDimX,
		typename RT::BlockDimY,
		typename RT::BlockDimZ,
		boost::mp11::mp_int<0>,
		typename CreateMetaMemoryManager<ParserConfigurationT, BaseActionT>::ReadOnlyBuffer*,
		const char*,
		const InputIndex*,
		ParsingError*,
		void**,
		const uint32_t
	>;

	ROB* m_d_rob;
	KernelLaunchConfiguration* m_launch_config;
	cudaStream_t m_stream;

	ParserKernel(KernelLaunchConfiguration* launch_config, cudaStream_t stream = 0)
		: m_launch_config(launch_config), m_stream(stream)
	{
		cudaMalloc(&m_d_rob, sizeof(ROB));
		ROB rob;
		M3::FillReadOnlyBuffer(rob, m_launch_config);
		cudaMemcpyAsync(m_d_rob, &rob, sizeof(ROB), cudaMemcpyHostToDevice, m_stream);
	}

	void Run(
		const char* input,
		const InputIndex* indices,
		ParsingError* errors,
		void** d_outputs, // Device array of pointers to device outputs
		const uint32_t count,
		void** h_outputs // Host array of pointers to device outputs
	)
	{
		constexpr int GROUP_SIZE = RT::WorkGroupSize::value;
		constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
		const unsigned int BLOCKS_COUNT = (count + GROUP_COUNT - 1) / GROUP_COUNT;
		constexpr auto kernel_ptr = &_parser_kernel<ParserConfigurationT, BaseActionT>;
		Launcher l(kernel_ptr);
		l(BLOCKS_COUNT, m_stream)(
			m_d_rob,
			input,
			indices,
			errors,
			d_outputs,
			count
		);

		using Actions = ActionIterator<BaseActionT>;

		using PostKernelHooks = boost::mp11::mp_filter<
			HavePostKernelHook,
			Actions
		>;

		boost::mp11::mp_for_each<PostKernelHooks>([&](auto action) {
			decltype(action)::PostKernelHook(*this, count, h_outputs);
		});
	}

	~ParserKernel()
	{
		cudaFree(m_d_rob);
	}

	static thrust::host_vector<uint64_t> OutputSizes()
	{
		thrust::host_vector<uint64_t> result;
		boost::mp11::mp_for_each<typename OC::RequestList>([&](auto i){
			using Request = decltype(i);
			result.push_back(sizeof(typename Request::OutputType));
		});
		return std::move(result);
	}
};

template<class ParserConfigurationT, class BaseActionT>
	/// <summary>
	/// Main kernel responsible for parsing json.
	/// </summary>
	/// <param name="block_data">Read-only data stored in shared memory.</param>
	/// <param name="input">Pointer to input bytes array.</param>
	/// <param name="indices">Pointer to an array of indices of object beginings. Requires guard at the end! indices[count] == length(input)</param>
	/// <param name="err">Output array for error codes.</param>
	/// <param name="output">Pointer to array of pointers. Each points to distinct output array.</param>
	/// <param name="count">Number of objects.</param>
	/// <returns></returns>
__global__ void __launch_bounds__(1024, 2)
	_parser_kernel(
		typename CreateMetaMemoryManager<ParserConfigurationT, BaseActionT>::ReadOnlyBuffer* readOnlyBuffer,
		const char* input,
		const InputIndex* indices,
		ParsingError* err,
		void** output,
		const uint32_t count
	)
{
	using PK = ParserKernel<ParserConfigurationT, BaseActionT>;
	using KC = typename PK::KC;
	using RT = typename PK::RT;
	__shared__ typename PK::M3::SharedBuffers sharedBuffers;
	KC kc(readOnlyBuffer, sharedBuffers, input, indices, output);
	if (RT::InputId() >= count)
		return;
	ParsingError e = BaseActionT::Invoke(kc);
	if (RT::WorkerId() == 0)
		err[RT::InputId()] = e;
}

