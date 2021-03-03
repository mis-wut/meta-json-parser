#pragma once
#include <cub/cub.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <utility>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/byte_algorithms.h>
#include <meta_json_parser/memory_request.h>

namespace JsonParse
{
	constexpr char4 WHITESPACES = { 0x20, 0x9, 0x0A, 0x0D };
	constexpr char4 VALID_ENDING = { ']', '}', ',', '\0' };
	template<class WorkGroupSizeT>
	constexpr unsigned int FULL_WARP_MASK = (0x1u << WorkGroupSizeT::value) - 1;

	struct BitOr
	{
		template <typename T>
		__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
		{
			return a | b;
		}
	};

	struct BitAnd
	{
		template <typename T>
		__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
		{
			return a & b;
		}
	};

	template<typename T>
	__device__ INLINE_METHOD T& AccessFirst(T& t)
	{
		return t;
	}

	template<typename ...ArgsT>
	struct SetValueDispatch
	{
		static_assert(sizeof...(ArgsT) <= 1, "Number of variadic args cannot be greater than 1.");
		template<typename T>
		__device__ INLINE_METHOD void operator()(ArgsT &... args, T v)
		{
			AccessFirst(std::forward<ArgsT&>(args)...) = v;
		}
	};

	template<>
	struct SetValueDispatch<>
	{
		template<typename T>
		__device__ INLINE_METHOD void operator()(T v) { }
	};

	template<class OperationT, class WorkGroupSizeT>
	struct Reduce_impl
	{
		using WarpReduce = typename cub::template WarpReduce<OperationT, WorkGroupSizeT::value>;
		using WarpReduceStorage = typename WarpReduce::TempStorage;
		using WarpReduceRequest = MemoryRequest<
			boost::mp11::mp_int<sizeof(WarpReduceStorage)>,
			MemoryUsage::AtomicUsage
		>;
		WarpReduceStorage& storage;
		__device__ INLINE_METHOD Reduce_impl(WarpReduceStorage& _storage) : storage(_storage) {}

		template<class ReduceOperatorT>
		__device__ INLINE_METHOD OperationT Reduce(OperationT value, ReduceOperatorT op)
		{
			return WarpReduce(storage).Reduce(value, op);
		}

		template<class ReduceOperatorT>
		__device__ INLINE_METHOD OperationT Reduce(OperationT value, ReduceOperatorT op, int valid_items)
		{
			return WarpReduce(storage).Reduce(value, op, valid_items);
		}
	};

	template<class OperationT, class WorkGroupSizeT>
	using ReduceRequest = Reduce_impl<OperationT, WorkGroupSizeT>::WarpReduceRequest;

	template<class OperationT, class WorkGroupSizeT, class KernelContextT>
	__device__ INLINE_METHOD Reduce_impl<OperationT, WorkGroupSizeT> Reduce(KernelContextT& kc)
	{
		using R = Reduce_impl<OperationT, WorkGroupSizeT>;
		return R(kc.m3
			.template Receive<typename R::WarpReduceRequest>()
			.template Alias<typename R::WarpReduceStorage>()
		);
	}

	template<class OperationT, class WorkGroupSizeT>
	struct Scan_impl
	{
		using WarpScan = typename cub::template WarpScan<OperationT, WorkGroupSizeT::value>;
		using WarpScanStorage = typename WarpScan::TempStorage;
		using WarpScanRequest = MemoryRequest<
			boost::mp11::mp_int<sizeof(WarpScanStorage)>,
			MemoryUsage::AtomicUsage
		>;
		WarpScanStorage& storage;
		__device__ INLINE_METHOD Scan_impl(WarpScanStorage& _storage) : storage(_storage) {}

		__device__ INLINE_METHOD OperationT Broadcast(OperationT value, int sourceThread)
		{
			return WarpScan(storage).Broadcast(value, sourceThread);
		}
	};

	template<class OperationT, class WorkGroupSizeT>
	using ScanRequest = Scan_impl<OperationT, WorkGroupSizeT>::WarpScanRequest;

	template<class OperationT, class WorkGroupSizeT, class KernelContextT>
	__device__ INLINE_METHOD Scan_impl<OperationT, WorkGroupSizeT> Scan(KernelContextT& kc)
	{
		using R = Scan_impl<OperationT, WorkGroupSizeT>;
		return R(kc.m3
			.template Receive<typename R::WarpScanRequest>()
			.template Alias<typename R::WarpScanStorage>()
		);
	}

	template<class OutTypeT>
	using UnsignedIntegerOperationType = boost::mp11::mp_if_c<
		sizeof(OutTypeT) <= sizeof(uint32_t),
		uint32_t,
		uint64_t
	>;

	template<class OutTypeT, class WorkGroupSizeT>
	using UnsignedIntegerRequests = boost::mp11::mp_list<
		ReduceRequest<
			int,
			WorkGroupSizeT
		>,
		ReduceRequest<
			UnsignedIntegerOperationType<OutTypeT>,
			WorkGroupSizeT
		>,
		ScanRequest<
			int,
			WorkGroupSizeT
		>
	>;

	template<class OutTypeT, class WorkGroupSizeT, class KernelContextT>
	struct UnsignedIntegerParser
	{
		using KC = KernelContextT;
		using R = UnsignedIntegerRequests<OutTypeT, WorkGroupSizeT>;
		using WS = WorkGroupSizeT;
		using OP = UnsignedIntegerOperationType<OutTypeT>;
		using RT = KC::RT;
		static_assert(std::is_arithmetic_v<OutTypeT>, "OutTypeT must be arithmetic.");

		__device__ __forceinline__ UnsignedIntegerParser(KC& kc) : _kc(kc) {}
	private:
		KC& _kc;

		template<typename ...ArgsT>
		__device__ INLINE_METHOD ParsingError Parse(ArgsT& ...args)
		{
			static_assert(WorkGroupSizeT::value >= 2, "WorkGroup must have a size of at least 2.");
			if (_kc.wgr.PeekChar(0) == '0')
			{
				char c = _kc.wgr.PeekChar(1);
				if (HasThisByte(WHITESPACES, c) ||
					HasThisByte(VALID_ENDING, c))
				{
					_kc.wgr.AdvanceBy(1);
					if (KC::RT::WorkerId() == 0)
						SetValueDispatch<ArgsT&...>()(args..., OutTypeT(0));
					return ParsingError::None;
				}
				return ParsingError::Other;
			}
			OutTypeT result = OutTypeT(1);
			char c = _kc.wgr.CurrentChar();
			bool isEnd = HasThisByte(WHITESPACES, c) || HasThisByte(VALID_ENDING, c);
			int activeThreads;
			activeThreads = Reduce<int, WS>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSizeT::value), cub::Min());
			activeThreads = Scan<int, WS>(_kc).Broadcast(activeThreads, 0);
			if (activeThreads == 0)
				return ParsingError::Other;
			if (activeThreads == WorkGroupSizeT::value)
			{
				//Unsupported uints larger than (GroupSize - 1) chars
				return ParsingError::Other;
			}
			int valid = c >= '0' && c <= '9';
			valid = Reduce<int, WS>(_kc).Reduce(valid || RT::WorkerId() >= activeThreads, BitAnd());
			valid = Scan<int, WS>(_kc).Broadcast(valid, 0);
			if (!valid)
				return ParsingError::Other;
			if (RT::WorkerId() < activeThreads)
			{
				//TODO
				//OutTypeT = static_cast<OutTypeT>(HERE SHOULD BE ACCESS TO PRECALCULATED VALUE);

				//Temporary solution, for sake of correctness
				OutTypeT power = 1;
				for (int i = 0; i < (activeThreads - RT::WorkerId() - 1); ++i)
					power *= OutTypeT(10);
				result *= static_cast<OutTypeT>(c - '0') * power;
			}
			result = Reduce<OP, WS>(_kc).Reduce(result, cub::Sum(), activeThreads);
			_kc.wgr.AdvanceBy(activeThreads);
			if (KC::RT::WorkerId() == 0)
				SetValueDispatch<ArgsT&...>()(args..., result);
			return ParsingError::None;
		}
	public:
		__device__ __forceinline__ ParsingError operator()()
		{
			return Parse();
		}

		__device__ __forceinline__ ParsingError operator()(OutTypeT& result)
		{
			return Parse(result);
		}
	};

	template<class OutTypeT, class WorkGroupSizeT>
	struct UnsignedInteger
	{
		static_assert(std::is_arithmetic_v<OutTypeT>, "OutTypeT must be arithmetic.");

		template<class KernelContextT>
		__device__ __forceinline__ static UnsignedIntegerParser<OutTypeT, WorkGroupSizeT, KernelContextT> KC(KernelContextT& kc)
		{
			return UnsignedIntegerParser<OutTypeT, WorkGroupSizeT, KernelContextT>(kc);
		}
	};
}