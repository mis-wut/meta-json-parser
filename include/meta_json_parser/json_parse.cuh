#pragma once
#include <utility>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <meta_json_parser/config.h>
#include <meta_json_parser/meta_utility/length_representation.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/byte_algorithms.h>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/parse.cuh>
#include <meta_json_parser/cub_wrapper.cuh>
#include <meta_json_parser/kernel_launch_configuration.cuh>

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

		static void __host__ Fill(Buffer& buffer, const KernelLaunchConfiguration* _)
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

	template<class KernelContextT, class CallbackFnT>
	__device__ __forceinline__ ParsingError __impl_WS_8_plus_Boolean(KernelContextT& _kc, CallbackFnT&& fn)
	{
		using KC = KernelContextT;
		using RT = typename KC::RT;
		using WorkGroupSize = typename RT::WorkGroupSize;
		static_assert(WorkGroupSize::value >= 5, "WorkGroup must have a size of at least 5.");
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
		found = Reduce<int, WorkGroupSize>(_kc).Reduce(found, BitAnd());
		found = Scan<int, WorkGroupSize>(_kc).Broadcast(found, 0);
		if (found & 0x1)
		{
			_kc.wgr.AdvanceBy(4);
			fn(true);
			return ParsingError::None;
		}
		else if (found & 0x2)
		{
			_kc.wgr.AdvanceBy(5);
			fn(false);
			return ParsingError::None;
		}
		return ParsingError::Other;
	}

	template<class KernelContextT, class CallbackFnT>
	__device__ __forceinline__ ParsingError __impl_WS_4_Boolean(KernelContextT& _kc, CallbackFnT&& fn)
	{
		using KC = KernelContextT;
		using RT = typename KC::RT;
		using WorkGroupSize = typename RT::WorkGroupSize;
		static_assert(WorkGroupSize::value == 4, "WorkGroup must have a size equal to 4.");
		JsonKeywords& keywords = _kc.m3
				.template Receive<BooleanRequest>()
				.template Alias<JsonKeywords>();
		int found = 0x0;
		char c = _kc.wgr.CurrentChar();
		{
			int isTrue = 1;
			int isFalse = 1;
			isTrue = c == keywords.words._true[RT::WorkerId()];
			isFalse = c == keywords.words._false[RT::WorkerId()];
			_kc.wgr.AdvanceBy(4);
			c = _kc.wgr.CurrentChar();
			if (RT::WorkerId() == 0)
			{
				isTrue |= HasThisByte(VALID_ENDING, c) || HasThisByte(WHITESPACES, c);
				isFalse |= c == 'e';
			}
			if (RT::WorkerId() == 1)
				isFalse |= HasThisByte(VALID_ENDING, c) || HasThisByte(WHITESPACES, c);
			found |= isTrue ? 0x1 : 0x0;
			found |= isFalse ? 0x2 : 0x0;
		}
		found = Reduce<int, WorkGroupSize>(_kc).Reduce(found, BitAnd());
		found = Scan<int, WorkGroupSize>(_kc).Broadcast(found, 0);
		if (found & 0x1)
		{
			//No need for advance
			fn(true);
			return ParsingError::None;
		}
		else if (found & 0x2)
		{
			_kc.wgr.AdvanceBy(1);
			fn(false);
			return ParsingError::None;
		}
		return ParsingError::Other;
	}

	template<int WorkGroupSize>
	struct __dispatch_impl_Boolean {
		template<class KernelContextT, class CallbackFnT>
		static __device__ __forceinline__ ParsingError __impl_Boolean(KernelContextT& _kc, CallbackFnT&& fn) {
			return __impl_WS_8_plus_Boolean(_kc, std::forward<CallbackFnT&&>(fn));
		}
	};

	template<>
	struct __dispatch_impl_Boolean<4> {
		template<class KernelContextT, class CallbackFnT>
		static __device__ __forceinline__ ParsingError __impl_Boolean(KernelContextT& _kc, CallbackFnT&& fn) {
			return __impl_WS_4_Boolean(_kc, std::forward<CallbackFnT&&>(fn));
		}
	};

	template<class KernelContextT, class CallbackFnT>
	__device__ INLINE_METHOD ParsingError Boolean(KernelContextT& _kc, CallbackFnT&& fn)
	{
		using KC = KernelContextT;
		using RT = typename KC::RT;
		using WorkGroupSize = typename RT::WorkGroupSize;
		return __dispatch_impl_Boolean<WorkGroupSize::value>::template __impl_Boolean(_kc, std::forward<CallbackFnT&&>(fn));
	}

    using IsNullRequest = FilledMemoryRequest<
        JsonKeywords::Buffer::Size,
        JsonKeywords,
        MemoryUsage::ReadOnly,
        MemoryType::Shared
    >;

    using IsNullRequests = boost::mp11::mp_list<
        IsNullRequest,
        ReduceRequest<
            int
        >,
        ScanRequest<
            int
        >
    >;

    template<class KernelContextT, class CallbackFnT>
    __device__ __forceinline__ ParsingError __impl_WS_8_plus_IsNull(KernelContextT& _kc, CallbackFnT&& fn)
    {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;
        static_assert(WorkGroupSize::value >= 5, "WorkGroup must have a size of at least 5.");
        JsonKeywords& keywords = _kc.m3
            .template Receive<IsNullRequest>()
            .template Alias<JsonKeywords>();
        char c = _kc.wgr.CurrentChar();
        int isNull = 1;
        if (RT::WorkerId() < 4)
            isNull = c == keywords.words._null[RT::WorkerId()];
        else if (RT::WorkerId() == 4)
            isNull = HasThisByte(VALID_ENDING, c) || HasThisByte(WHITESPACES, c);
        //TODO function should return an error for input like 'nullX', 'null3' etc.
        isNull = Reduce<int, WorkGroupSize>(_kc).Reduce(isNull, LogicalAnd());
        isNull = Scan<int, WorkGroupSize>(_kc).Broadcast(isNull, 0);
        if (isNull)
            _kc.wgr.AdvanceBy(4);
        fn(isNull);
        return ParsingError::None;
    }

    template<class KernelContextT, class CallbackFnT>
    __device__ __forceinline__ ParsingError __impl_WS_4_IsNull(KernelContextT& _kc, CallbackFnT&& fn)
    {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;
        static_assert(WorkGroupSize::value == 4, "WorkGroup must have a size equal to 4.");
        JsonKeywords& keywords = _kc.m3
            .template Receive<IsNullRequest>()
            .template Alias<JsonKeywords>();
        char c = _kc.wgr.CurrentChar();
        int isNull = c == keywords.words._null[RT::WorkerId()];
        isNull = Reduce<int, WorkGroupSize>(_kc).Reduce(isNull, LogicalAnd());
        isNull = Scan<int, WorkGroupSize>(_kc).Broadcast(isNull, 0);
        // If null was mismatch at this point we pass false to callback fn.
        // In case of mismatch null we cannot AdvanceBy because we do not wan't to skip it.
        if (!isNull) {
            fn(false);
            return ParsingError::None;
        }
        _kc.wgr.AdvanceBy(4);
        if (RT::WorkerId() == 0)
            isNull = isNull && (HasThisByte(VALID_ENDING, c) || HasThisByte(WHITESPACES, c));
        isNull = Scan<int, WorkGroupSize>(_kc).Broadcast(isNull, 0);
        if (!isNull)
            return ParsingError::Other;
        fn(true);
        return ParsingError::None;
    }

    template<int WorkGroupSize>
    struct __dispatch_impl_IsNull {
        template<class KernelContextT, class CallbackFnT>
        static __device__ __forceinline__ ParsingError __impl_IsNull(KernelContextT& _kc, CallbackFnT&& fn) {
            return __impl_WS_8_plus_IsNull(_kc, std::forward<CallbackFnT&&>(fn));
        }
    };

    template<>
    struct __dispatch_impl_IsNull<4> {
        template<class KernelContextT, class CallbackFnT>
        static __device__ __forceinline__ ParsingError __impl_IsNull(KernelContextT& _kc, CallbackFnT&& fn) {
            return __impl_WS_4_IsNull(_kc, std::forward<CallbackFnT&&>(fn));
        }
    };

    /**
     * Checks if current input contains 'null' that ends with valid character. If input equals to 'null', then input
     * will be advanced till its end (by 4 characters) and 'true' will be passed to callback function. Otherwise,
     * input will not be advanced and 'false' will be passed to callback function.
     * @tparam KernelContextT
     * @tparam CallbackFnT
     * @param _kc Kernel context
     * @param fn Callback function
     * @return
     */
    template<class KernelContextT, class CallbackFnT>
    __device__ INLINE_METHOD ParsingError IsNull(KernelContextT& _kc, CallbackFnT&& fn)
    {
        using KC = KernelContextT;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;
        return __dispatch_impl_IsNull<WorkGroupSize::value>::template __impl_IsNull(_kc, std::forward<CallbackFnT&&>(fn));
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

	template<class OutTypeT, class KernelContextT, class CallbackFnT>
	__device__ INLINE_METHOD ParsingError UnsignedInteger(KernelContextT& _kc, CallbackFnT&& fn)
	{
		static_assert(std::is_arithmetic_v<OutTypeT>, "OutTypeT must be arithmetic.");
		using KC = KernelContextT;
		using R = UnsignedIntegerRequests<OutTypeT>;
		using RT = typename KC::RT;
		using WorkGroupSize = typename RT::WorkGroupSize;
		using OP = UnsignedIntegerOperationType<OutTypeT>;
		static_assert(WorkGroupSize::value >= 2, "WorkGroup must have a size of at least 2.");

		if (_kc.wgr.PeekChar(0) == '0')
		{
			char c = _kc.wgr.PeekChar(1);
			if (HasThisByte(WHITESPACES, c) ||
				HasThisByte(VALID_ENDING, c))
			{
				_kc.wgr.AdvanceBy(1);
				if (KC::RT::WorkerId() == 0)
					fn(OutTypeT(0));
				return ParsingError::None;
			}
			return ParsingError::Other;
		}
		OutTypeT output = OutTypeT(0);

		int activeThreads;
		//First loop
		{
			char c = _kc.wgr.CurrentChar();
			bool isEnd = HasThisByte(WHITESPACES, c) || HasThisByte(VALID_ENDING, c);
			activeThreads = Reduce<int, WorkGroupSize>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSize::value), cub::Min());
			activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
			if (activeThreads == 0)
				return ParsingError::Other;
			int valid = c >= '0' && c <= '9';
			valid = Reduce<int, WorkGroupSize>(_kc).Reduce(valid || RT::WorkerId() >= activeThreads, BitAnd());
			valid = Scan<int, WorkGroupSize>(_kc).Broadcast(valid, 0);
			if (!valid)
				return ParsingError::Other;
			OutTypeT power = OutTypeT(1);
			if (RT::WorkerId() < activeThreads)
			{
				//TODO
				//OutTypeT = static_cast<OutTypeT>(HERE SHOULD BE ACCESS TO PRECALCULATED VALUE);

				//Temporary solution, for sake of correctness
				power = 1;
				for (int i = 0; i < (activeThreads - RT::WorkerId() - 1); ++i)
					power *= OutTypeT(10);
				power = static_cast<OutTypeT>(c - '0') * power;
			}
			output += Reduce<OP, WorkGroupSize>(_kc).Reduce(power, cub::Sum(), activeThreads);
			_kc.wgr.AdvanceBy(activeThreads);
		}

		while (activeThreads == WorkGroupSize::value)
		{
			char c = _kc.wgr.CurrentChar();
			bool isEnd = HasThisByte(WHITESPACES, c) || HasThisByte(VALID_ENDING, c);
			activeThreads = Reduce<int, WorkGroupSize>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSize::value), cub::Min());
			activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
			// No digits/charactes in each next workgroup pass
			if (activeThreads == 0)
				break;
			//Temporary solution
			for (int i = 0; i < activeThreads; ++i)
				output *= OutTypeT(10);
			int valid = c >= '0' && c <= '9';
			valid = Reduce<int, WorkGroupSize>(_kc).Reduce(valid || RT::WorkerId() >= activeThreads, BitAnd());
			valid = Scan<int, WorkGroupSize>(_kc).Broadcast(valid, 0);
			if (!valid)
				return ParsingError::Other;
			OutTypeT power = OutTypeT(1);
			if (RT::WorkerId() < activeThreads)
			{
				//TODO
				//OutTypeT = static_cast<OutTypeT>(HERE SHOULD BE ACCESS TO PRECALCULATED VALUE);

				//Temporary solution, for sake of correctness
				power = 1;
				for (int i = 0; i < (activeThreads - RT::WorkerId() - 1); ++i)
					power *= OutTypeT(10);
				power = static_cast<OutTypeT>(c - '0') * power;
			}
			output += Reduce<OP, WorkGroupSize>(_kc).Reduce(power, cub::Sum(), activeThreads);
			_kc.wgr.AdvanceBy(activeThreads);
		}

		if (KC::RT::WorkerId() == 0)
			fn(output);
		return ParsingError::None;
	}

    template<class OutTypeT, class KernelContextT, class CallbackFnT>
    __device__ INLINE_METHOD ParsingError SignedInteger(KernelContextT& _kc, CallbackFnT&& fn)
    {
        static_assert(std::is_arithmetic_v<OutTypeT>, "OutTypeT must be arithmetic.");
        using KC = KernelContextT;
        using R = UnsignedIntegerRequests<OutTypeT>;
        using RT = typename KC::RT;
        using WorkGroupSize = typename RT::WorkGroupSize;
        using OP = UnsignedIntegerOperationType<OutTypeT>;
        static_assert(WorkGroupSize::value >= 2, "WorkGroup must have a size of at least 2.");

        bool minus = false;
        if (_kc.wgr.PeekChar(0) == '-') {
            minus = true;
            _kc.wgr.AdvanceBy(1);
        }
        return UnsignedInteger<OutTypeT>(_kc, [&minus, &fn](auto result) { return fn(minus ? -result : result); } );
    }

    template<bool Signed = true>
    struct IntegerFunctionDispatch {
        template<class OutTypeT, class KernelContextT, class CallbackFnT>
        static __device__ __forceinline__ ParsingError function(KernelContextT& _kc, CallbackFnT&& fn) {
            return SignedInteger<OutTypeT>(_kc, fn);
        }
    };

    template<>
    struct IntegerFunctionDispatch<false> {
        template<class OutTypeT, class KernelContextT, class CallbackFnT>
        static __device__ __forceinline__ ParsingError function(KernelContextT& _kc, CallbackFnT&& fn) {
            return UnsignedInteger<OutTypeT>(_kc, fn);
        }
    };

    template<class SignedT, class OutTypeT, class KernelContextT, class CallbackFnT>
    __device__ INLINE_METHOD ParsingError Integer(KernelContextT& _kc, CallbackFnT&& fn) {
        return IntegerFunctionDispatch<SignedT::value>::template function<OutTypeT>(_kc, fn);
    }

	template<class OutTypeT, class KernelContextT, class CallbackFnT>
	__device__ INLINE_METHOD ParsingError UnsignedIntegerFitsWorkgroup(KernelContextT& _kc, CallbackFnT&& fn)
	{
		static_assert(std::is_arithmetic_v<OutTypeT>, "OutTypeT must be arithmetic.");
		using KC = KernelContextT;
		using R = UnsignedIntegerRequests<OutTypeT>;
		using RT = typename KC::RT;
		using WorkGroupSize = typename RT::WorkGroupSize;
		using OP = UnsignedIntegerOperationType<OutTypeT>;
		static_assert(WorkGroupSize::value >= 2, "WorkGroup must have a size of at least 2.");

		if (_kc.wgr.PeekChar(0) == '0')
		{
			char c = _kc.wgr.PeekChar(1);
			if (HasThisByte(WHITESPACES, c) ||
				HasThisByte(VALID_ENDING, c))
			{
				_kc.wgr.AdvanceBy(1);
				if (KC::RT::WorkerId() == 0)
					fn(OutTypeT(0));
				return ParsingError::None;
			}
			return ParsingError::Other;
		}
		OutTypeT result = OutTypeT(1);
		char c = _kc.wgr.CurrentChar();
		bool isEnd = HasThisByte(WHITESPACES, c) || HasThisByte(VALID_ENDING, c);
		int activeThreads;
		activeThreads = Reduce<int, WorkGroupSize>(_kc).Reduce((isEnd ? RT::WorkerId() : WorkGroupSize::value), cub::Min());
		activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
		if (activeThreads == 0)
			return ParsingError::Other;
		// It should not occur as integer should fit in work group
		//if (activeThreads == WorkGroupSize::value)
		//{
		//	return ParsingError::Other;
		//}
		int valid = c >= '0' && c <= '9';
		valid = Reduce<int, WorkGroupSize>(_kc).Reduce(valid || RT::WorkerId() >= activeThreads, BitAnd());
		valid = Scan<int, WorkGroupSize>(_kc).Broadcast(valid, 0);
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
		result = Reduce<OP, WorkGroupSize>(_kc).Reduce(result, cub::Sum(), activeThreads);
		_kc.wgr.AdvanceBy(activeThreads);
		if (KC::RT::WorkerId() == 0)
			fn(result);
		return ParsingError::None;
	}

	using StringRequests = boost::mp11::mp_list<
		ReduceRequest<
			int
		>,
		ScanRequest<
			int
		>
	>;

	template<class KernelContextT, class CallbackFnT>
	__device__ INLINE_METHOD ParsingError String(KernelContextT& _kc, CallbackFnT&& fn)
	{
		using KC = KernelContextT;
		using R = StringRequests;
		using RT = typename KC::RT;
		using WorkGroupSize = typename RT::WorkGroupSize;

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
			activeThreads = Reduce<int, WorkGroupSize>(_kc)
				.Reduce((!isEscaped && c == '"') ? RT::WorkerId() : WorkGroupSize::value, cub::Min());
			activeThreads = Scan<int, WorkGroupSize>(_kc).Broadcast(activeThreads, 0);
			if (!_kc.wgr.all_sync(RT::WorkerId() > activeThreads || c >= CHARS_FROM))
				return ParsingError::Other;
			//Call function after validation
			ParsingError err = fn(isEscaped, activeThreads);
			if (err != ParsingError::None)
				return err;
			//Advance
			if (activeThreads != WorkGroupSize::value)
			{
				_kc.wgr.AdvanceBy(activeThreads + 1);
				break;
			}
			previousBackSlashes = Scan<int, WorkGroupSize>(_kc)
				.Broadcast(previousBackSlashes, WorkGroupSize::value - 1);
			_kc.wgr.AdvanceBy(WorkGroupSize::value);
		} while (true);
		return ParsingError::None;
	}
}
