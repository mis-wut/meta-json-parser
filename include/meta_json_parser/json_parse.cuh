#pragma once
#include <utility>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/byte_algorithms.h>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/parse.cuh>
#include <meta_json_parser/cub_wrapper.cuh>

namespace JsonParse
{
	constexpr char4 WHITESPACES = Parse::WHITESPACES;
	constexpr char4 ESCAPEABLE_1 = { '"', '\\', '/', 'b' };
	constexpr char4 ESCAPEABLE_2 = { 'f', 'n', 'r', 't' };
	constexpr char4 VALID_ENDING = { ']', '}', ',', '\0' };
	static constexpr char CHARS_FROM = 0x20;

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


	template<typename FuncT = void, typename ...ArgsT>
	__device__ INLINE_METHOD auto CallFunctionDispatch(...)
		-> decltype(ParsingError())
	{
		return ParsingError::None;
	}

	template<typename FuncT, typename ...ArgsT>
	__device__ INLINE_METHOD auto CallFunctionDispatch(FuncT& fun, ArgsT&... args)
		-> decltype(fun(std::forward<ArgsT&>(args)...), ParsingError())
	{
		return fun(std::forward<ArgsT&>(args)...);
	}

	struct JsonKeywords
	{
		struct Words
		{
			char _null[4];
			char _true[4];
			char _false[5];
			char _unused[3];
		} words;
		//0x0 | n u l l t r u e
		//0x8 | f a l s e . . .
		using Buffer = StaticBuffer_c<sizeof(Words)>;

		static void __host__ Fill(Buffer& buffer)
		{
			auto buf = "nulltruefalse\0\0\0";
			std::copy_n(buf, sizeof(Words), buffer.data);
		}
	};

	using BooleanRequest = FilledMemoryRequest<
		JsonKeywords::Buffer::Size,
		JsonKeywords,
		MemoryUsage::ReadOnly,
		MemoryType::Shared
	>;

	using BooleanRequests = boost::mp11::mp_list<
		BooleanRequest,
		ReduceRequest<
			int
		>,
		ScanRequest<
			int
		>
	>;

	template<class WorkGroupSizeT, class KernelContextT, class ...ArgsT>
	__device__ INLINE_METHOD ParsingError Boolean(KernelContextT& _kc, ArgsT& ...args)
	{
		using KC = KernelContextT;
		using WS = WorkGroupSizeT;
		using RT = typename KC::RT;
		static_assert(WorkGroupSizeT::value >= 5, "WorkGroup must have a size of at least 5.");
		JsonKeywords& keywords = _kc.m3
				.template Receive<BooleanRequest>()
				.template Alias<JsonKeywords>();
		int found = 0x0;
		char c = _kc.wgr.CurrentChar();
		{
			int isTrue = 1;
			if (RT::WorkerId() < 4)
				isTrue = c == keywords.words._true[RT::WorkerId()];
			else if (RT::WorkerId() == 4)
				isTrue = HasThisByte(VALID_ENDING, c) || HasThisByte(WHITESPACES, c);
			found |= isTrue ? 0x1 : 0x0;
		}
		{
			int isFalse = 1;
			if (RT::WorkerId() < 5)
				isFalse = c == keywords.words._false[RT::WorkerId()];
			else if (RT::WorkerId() == 5)
				isFalse = HasThisByte(VALID_ENDING, c) || HasThisByte(WHITESPACES, c);
			found |= isFalse ? 0x2 : 0x0;
		}
		found = Reduce<int, WS>(_kc).Reduce(found, BitAnd());
		found = Scan<int, WS>(_kc).Broadcast(found, 0);
		if (found & 0x1)
		{
			_kc.wgr.AdvanceBy(4);
			SetValueDispatch<ArgsT&...>()(args..., true);
			return ParsingError::None;
		}
		else if (found & 0x2)
		{
			_kc.wgr.AdvanceBy(5);
			SetValueDispatch<ArgsT&...>()(args..., false);
			return ParsingError::None;
		}
		return ParsingError::Other;
	}

	template<class OutTypeT>
	using UnsignedIntegerOperationType = boost::mp11::mp_if_c<
		sizeof(OutTypeT) <= sizeof(uint32_t),
		uint32_t,
		uint64_t
	>;

	template<class OutTypeT>
	using UnsignedIntegerRequests = boost::mp11::mp_list<
		ReduceRequest<
			int
		>,
		ReduceRequest<
			UnsignedIntegerOperationType<OutTypeT>
		>,
		ScanRequest<
			int
		>
	>;

	template<class OutTypeT, class WorkGroupSizeT, class KernelContextT>
	struct UnsignedIntegerParser
	{
		using KC = KernelContextT;
		using R = UnsignedIntegerRequests<OutTypeT>;
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

	using StringRequests = boost::mp11::mp_list<
		ReduceRequest<
			int
		>,
		ScanRequest<
			int
		>
	>;

	template<class WorkGroupSizeT, class KernelContextT>
	struct StringParser
	{
		using KC = KernelContextT;
		using R = StringRequests;
		using WS = WorkGroupSizeT;
		//using OP = UnsignedIntegerOperationType<OutTypeT>;
		using RT = KC::RT;

		__device__ __forceinline__ StringParser(KC& kc) : _kc(kc) {}
	private:
		KC& _kc;

		template<typename ...ArgsT>
		__device__ INLINE_METHOD ParsingError Parse(ArgsT& ...args)
		{
			if (_kc.wgr.PeekChar(0) != '"')
				return ParsingError::Other;
			_kc.wgr.AdvanceBy(1);
			constexpr unsigned int previousActiveBackSlash = 0x7F'FF'FF'FFu;
			constexpr unsigned int previousNoBackSlash = 0xFF'FF'FF'FFu;
			unsigned int previousBackSlashes = previousNoBackSlash;
			do
			{
				unsigned char c = _kc.wgr.CurrentChar();
				//Example, for 16 bits:
				//notSlash = 0011'1011'1000'1001, for an example: "a\\b\\\nLK\rAB\\"
				//bit set for each no backslash
				uint32_t notSlash = ~_kc.wgr.ballot_sync(c == '\\');
				//before =
				//  t[0] = 1111'1111'1111'1111,  t[8] = 1000'1001'1111'1111
				//  t[1] = 1111'1111'1111'1111,  t[9] = 1100'0100'1111'1111
				//  t[2] = 0111'1111'1111'1111,  t[A] = 1110'0010'0111'1111
				//  t[3] = 0011'1111'1111'1111,  t[B] = 0111'0001'0011'1111
				//  t[4] = 1001'1111'1111'1111,  t[C] = 1011'1000'1001'1111
				//  t[5] = 0100'1111'1111'1111,  t[D] = 1101'1100'0100'1111
				//  t[6] = 0010'0111'1111'1111,  t[E] = 1110'1110'0010'0111
				//  t[7] = 0001'0011'1111'1111,  t[F] = 0111'0111'0001'0011
				//each zero indicates backslash, bits are shifted to appropriate positions
				uint32_t before = __funnelshift_lc(previousBackSlashes, notSlash, RT::GroupSize() - RT::WorkerId());
				//consecutiveZeros =
				//  t[0] = 0,  t[8] = 0
				//  t[1] = 0,  t[9] = 0
				//  t[2] = 1,  t[A] = 0
				//  t[3] = 2,  t[B] = 1
				//  t[4] = 0,  t[C] = 0
				//  t[5] = 1,  t[D] = 0
				//  t[6] = 2,  t[E] = 0
				//  t[7] = 3,  t[F] = 1
				//length of preceding backslashes sequence
				uint32_t consecutiveZeros = __clz(before << (32 - RT::GroupSize()));
				//If there are odd number of backslashes before, it is escaped
				bool isEscaped = consecutiveZeros & 0x1;
				previousBackSlashes = (!isEscaped && c == '\\') ? previousActiveBackSlash : previousNoBackSlash;
				// Test if escapes are valid
				{
					bool valid = !isEscaped || HasThisByte(ESCAPEABLE_1, c) || HasThisByte(ESCAPEABLE_2, c);
					//TODO hex support
					if (!_kc.wgr.all_sync(valid))
						return ParsingError::Other;
				}
				int activeThreads;
				activeThreads = Reduce<int, WS>(_kc).Reduce((!isEscaped && c == '"') ? RT::WorkerId() : WS::value, cub::Min());
				activeThreads = Scan<int, WS>(_kc).Broadcast(activeThreads, 0);
				if (!_kc.wgr.all_sync(RT::WorkerId() > activeThreads || c >= CHARS_FROM))
					return ParsingError::Other;
				//Call function after validation
				ParsingError err = CallFunctionDispatch(std::forward<ArgsT&>(args)..., isEscaped, activeThreads);
				if (err != ParsingError::None)
					return err;
				//Advance
				if (activeThreads != WS::value)
				{
					_kc.wgr.AdvanceBy(activeThreads + 1);
					break;
				}
				previousBackSlashes = Scan<int, WS>(_kc).Broadcast(previousBackSlashes, WS::value - 1);
				_kc.wgr.AdvanceBy(WS::value);
			} while (true);
			return ParsingError::None;
		}
	public:
		template<class ...ArgsT>
		__device__ __forceinline__ ParsingError operator()(ArgsT& ...args)
		{
			return Parse(std::forward<ArgsT&>(args)...);
		}
	};

	template<class WorkGroupSizeT>
	struct String
	{
		template<class KernelContextT>
		__device__ __forceinline__ static StringParser<WorkGroupSizeT, KernelContextT> KC(KernelContextT& kc)
		{
			return StringParser<WorkGroupSizeT, KernelContextT>(kc);
		}
	};
}