#pragma once
#include <boost/mp11/list.hpp>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/kernel_launcher.cuh>
#include <cstdint>

template<class ParserConfigurationT, class BaseActionT>
__global__ void __launch_bounds__(1024, 2)
_parser_kernel(
	MetaMemoryManager<ParserConfigurationT>::ReadOnlyBuffer* readOnlyBuffer,
	const char* input,
	const InputIndex* indices,
	ParsingError* err,
	void** output,
	const uint32_t count);

template<class ParserConfigurationT, class BaseActionT>
struct ParserKernel
{
	using M3 = MetaMemoryManager<ParserConfigurationT>;
	using KC = KernelContext<ParserConfigurationT>;
	using RT = ParserConfigurationT::RuntimeConfiguration;
	using Launcher = KernelLauncherFixedResources<
		RT::BlockDimX,
		RT::BlockDimY,
		RT::BlockDimZ,
		boost::mp11::mp_int<0>,
		typename MetaMemoryManager<ParserConfigurationT>::ReadOnlyBuffer*,
		const char*,
		const InputIndex*,
		ParsingError*,
		void**,
		const uint32_t
	>;
	static const Launcher kernel;
};

template<class P, class A>
const ParserKernel<P, A>::Launcher ParserKernel<P, A>::kernel(_parser_kernel<P, A>);

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
		typename MetaMemoryManager<ParserConfigurationT>::ReadOnlyBuffer* readOnlyBuffer,
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
	//TODO fill shared buffers
	KC context(sharedBuffers, input, indices);
}

