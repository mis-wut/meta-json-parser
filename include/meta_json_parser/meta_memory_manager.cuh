#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/function.hpp>
#include <boost/mp11/bind.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/integer_sequence.hpp>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/static_buffer.h>
#include <meta_json_parser/byte_algorithms.h>
#include <utility>

template<class ParserConfigurationT>
struct MetaMemoryManager
{
	static_assert(std::is_same_v<
			boost::mp11::mp_rename<ParserConfigurationT, ParserConfiguration>,
			ParserConfigurationT
		>,
		"MetaMemoryManager takes ParserConfiguration as its input");
};

template<class ...ParserConfigurationArgsT>
struct MetaMemoryManager<ParserConfiguration<ParserConfigurationArgsT...>>
{
	using _ParserConfiguration = ParserConfiguration<ParserConfigurationArgsT...>;
	using RT = _ParserConfiguration::RuntimeConfiguration;

	struct IntPlusMemoryRequest
	{
		template<class Accumulator, class Request>
		using fn = boost::mp11::mp_int<Accumulator::value + GetRequestSize<Request, RT>::value>;
	};

	using _MemoryConfiguration = typename _ParserConfiguration::MemoryConfiguration;

	template<class ListT>
	using SumRequests = boost::mp11::mp_fold_q<
		ListT,
		boost::mp11::mp_int<0>,
		IntPlusMemoryRequest
	>;

	template<class ListT>
	using OnePerBlockSize = SumRequests<ListT>;

	template<class ListT>
	using OnePerGroupSize = boost::mp11::mp_mul<
		SumRequests<ListT>,
		typename _ParserConfiguration::RuntimeConfiguration::WorkGroupCount
	>;

	using ReadOnlyBuffer = StaticBuffer<OnePerBlockSize<_MemoryConfiguration::ReadOnlyList>>;
	using ActionBuffer = StaticBuffer<OnePerGroupSize<_MemoryConfiguration::ActionList>>;
	using AtomicBuffer = StaticBuffer<OnePerGroupSize<_MemoryConfiguration::AtomicList>>;

	//TODO now only shared memory is supported, add support for global and constant
	//TODO take alignment into account
	struct SharedBuffers {
		ReadOnlyBuffer __align__(16) readOnlyBuffer;
		ActionBuffer __align__(16) actionBuffer;
		AtomicBuffer __align__(16) atomicBuffer;
	} &sharedBuffers;

	__device__ __forceinline__ MetaMemoryManager(
		SharedBuffers& pSharedBuffer,
		ReadOnlyBuffer* pInputBuffers)
		: sharedBuffers(pSharedBuffer)
	{
		uint32_t* out = reinterpret_cast<uint32_t*>(&sharedBuffers.readOnlyBuffer);
		uint32_t* it = reinterpret_cast<uint32_t*>(pInputBuffers);
		const uint32_t* end = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(it) + ReadOnlyBuffer::size);
		out += RT::WorkerInBlockId();
		it += RT::WorkerInBlockId();
		for (; it < end; it += RT::WorkersInBlock(), out += RT::WorkersInBlock())
		{
			*out = *it;
		}
		__syncthreads();
	}

	__host__ static void FillReadOnlyBuffer(ReadOnlyBuffer& readOnlyBuffer)
	{
		using ROL = typename _MemoryConfiguration::ReadOnlyList;
		boost::mp11::mp_for_each<boost::mp11::mp_iota<boost::mp11::mp_size<ROL>>>([&](auto i) {
			constexpr int I = decltype(i)::value;
			constexpr int OFFSET = SumRequests<boost::mp11::mp_take_c<ROL, I>>::value;
			using RQ = boost::mp11::mp_at_c<ROL, I>;
			using B = typename RQ::Buffer;
			RQ::FillFn::Fill(*reinterpret_cast<B*>(&readOnlyBuffer.data[OFFSET]));
		});
	}

	template<class MemoryUsageT>
	__host__ __device__ __forceinline__ decltype(auto) GetBuffer() { }

	template<>
	__host__ __device__ __forceinline__ decltype(auto) GetBuffer<MemoryUsage::ReadOnly>()
	{
		return std::forward<ReadOnlyBuffer&>(sharedBuffers.readOnlyBuffer);
	}

	template<>
	__host__ __device__ __forceinline__ decltype(auto) GetBuffer<MemoryUsage::ActionUsage>()
	{
		return std::forward<ActionBuffer&>(sharedBuffers.actionBuffer);
	}

	template<>
	__host__ __device__ __forceinline__ decltype(auto) GetBuffer<MemoryUsage::AtomicUsage>()
	{
		return std::forward<AtomicBuffer&>(sharedBuffers.atomicBuffer);
	}

	//TODO take alignment into account
	template<class MemoryRequestT>
	__host__ __device__ __forceinline__ decltype(auto) ReceiveForGroup(int pGroupId)
	{
		using _MemoryUsage = typename MemoryRequestT::MemoryUsage;
		using _List = boost::mp11::mp_second<boost::mp11::mp_map_find<
			boost::mp11::mp_list<
				boost::mp11::mp_list<MemoryUsage::ReadOnly, typename _MemoryConfiguration::ReadOnlyList>,
				boost::mp11::mp_list<MemoryUsage::ActionUsage, typename _MemoryConfiguration::ActionList>,
				boost::mp11::mp_list<MemoryUsage::AtomicUsage, typename _MemoryConfiguration::AtomicList>
			>,
			_MemoryUsage
		>>;
		using _Index = boost::mp11::mp_find<_List, MemoryRequestT>;
		static_assert(_Index::value < boost::mp11::mp_size<_List>::value, "Request is not present in MetaMemoryManager");
		using _AccFun = boost::mp11::mp_second<boost::mp11::mp_map_find<
			boost::mp11::mp_list<
				boost::mp11::mp_list<MemoryUsage::ReadOnly, boost::mp11::mp_quote<OnePerBlockSize>>,
				boost::mp11::mp_list<MemoryUsage::ActionUsage, boost::mp11::mp_quote<OnePerGroupSize>>,
				boost::mp11::mp_list<MemoryUsage::AtomicUsage, boost::mp11::mp_quote<OnePerGroupSize>>
			>,
			_MemoryUsage
		>>;
		using _Head = boost::mp11::mp_eval_if_c<
			_Index::value == 0,
			boost::mp11::mp_list<>,
			boost::mp11::mp_take,
			_List,
			boost::mp11::mp_int<static_cast<int>(_Index::value - 1)>
		>;
		using _BufferOffset = typename _AccFun::template fn<_Head>;
		using _Buffer = GetRequestBuffer<MemoryRequestT, RT>;
		_Buffer& buffer = OffsetBytesAs<_BufferOffset, _Buffer>(GetBuffer<_MemoryUsage>());
		return std::is_same_v<_MemoryUsage, MemoryUsage::ReadOnly>
			? std::forward<_Buffer&>((&buffer)[0])
			: std::forward<_Buffer&>((&buffer)[pGroupId]);
	}

	template<class MemoryRequestT>
	__device__ __forceinline__ decltype(auto) Receive()
	{
		return ReceiveForGroup<MemoryRequestT>(threadIdx.y);
	}
};
