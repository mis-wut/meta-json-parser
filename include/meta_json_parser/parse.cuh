#pragma once
#include <cuda_runtime_api.h>
#include <meta_json_parser/config.h>
#include <meta_json_parser/byte_algorithms.h>
#include <meta_json_parser/cub_wrapper.cuh>

namespace Parse
{
	constexpr char4 WHITESPACES = { 0x20, 0x9, 0x0A, 0x0D };

	namespace StopTag
	{
		struct StopAt {};
		struct StopAfter {};
	}

	using FindNoneWhiteRequests = boost::mp11::mp_list<
		ReduceRequest<
			int
		>,
		ScanRequest<
			int
		>
	>;

	template<class WorkGroupSizeT, class KernelContextT>
	struct FindNoneWhiteParser
	{
		using KC = KernelContextT;
		using R = FindNoneWhiteRequests;
		using WS = WorkGroupSizeT;
		using RT = typename KC::RT;

		static constexpr char CHARS_FROM = 0x20;

		__device__ __forceinline__ FindNoneWhiteParser(KC& kc) : _kc(kc) {}
	private:
		KC& _kc;
	public:
		template<class StopTagT>
		__device__ INLINE_METHOD ParsingError Do()
		{
			int activeThreads = 0;
			char activeChar;
			while (true)
			{
				activeChar = _kc.wgr.CurrentChar();
				bool white = HasThisByte(WHITESPACES, activeChar);
				activeThreads = Reduce<int, WS>(_kc).Reduce(white ? WS::value : threadIdx.x, cub::Min());
				activeThreads = Scan<int, WS>(_kc).Broadcast(activeThreads, 0);
				//There is non white char in activeThreads
				if (activeThreads != WS::value)
				{
					activeChar = _kc.wgr.PeekChar(activeThreads);
					if (activeChar < CHARS_FROM)
						return ParsingError::Other;
					_kc.wgr.AdvanceBy(activeThreads + (std::is_same_v<StopTagT, StopTag::StopAt> ? 0 : 1));
					return ParsingError::None;
				}
				_kc.wgr.AdvanceBy(WS::value);
			}
		}
	};

	template<class WorkGroupSizeT>
	struct FindNoneWhite
	{
		template<class KernelContextT>
		__device__ __forceinline__ static FindNoneWhiteParser<WorkGroupSizeT, KernelContextT> KC(KernelContextT& kc)
		{
			return FindNoneWhiteParser<WorkGroupSizeT, KernelContextT>(kc);
		}
	};

	template<char TO_FIND, class WorkGroupSizeT, class KernelContextT>
	struct FindNextParser
	{
		using KC = KernelContextT;
		using R = FindNoneWhiteRequests;
		using WS = WorkGroupSizeT;
		using RT = typename KC::RT;

		static constexpr char CHARS_FROM = 0x20;

		__device__ __forceinline__ FindNextParser(KC& kc) : _kc(kc) {}
	private:
		KC& _kc;
	public:
		template<class StopTagT>
		__device__ INLINE_METHOD ParsingError Do()
		{
			int activeThreads = 0;
			char activeChar;
			while (true)
			{
				activeChar = _kc.wgr.CurrentChar();
				bool white = HasThisByte(WHITESPACES, activeChar);
				activeThreads = Reduce<int, WS>(_kc).Reduce(white ? WS::value : threadIdx.x, cub::Min());
				activeThreads = Scan<int, WS>(_kc).Broadcast(activeThreads, 0);
				//There is non white char in activeThreads
				if (activeThreads != WS::value)
				{
					activeChar = _kc.wgr.PeekChar(activeThreads);
					if (activeChar == TO_FIND)
					{
						_kc.wgr.AdvanceBy(activeThreads + (std::is_same_v<StopTagT, StopTag::StopAt> ? 0 : 1));
						return ParsingError::None;
					}
					else if (activeChar < CHARS_FROM)
						return ParsingError::Other;
					else
						return ParsingError::Other;
				}
				_kc.wgr.AdvanceBy(WS::value);
			}
		}
	};

	template<char TO_FIND, class WorkGroupSizeT>
	struct FindNext
	{
		template<class KernelContextT>
		__device__ __forceinline__ static FindNextParser<TO_FIND, WorkGroupSizeT, KernelContextT> KC(KernelContextT& kc)
		{
			return FindNextParser<TO_FIND, WorkGroupSizeT, KernelContextT>(kc);
		}
	};
}
