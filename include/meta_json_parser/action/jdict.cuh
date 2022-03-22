#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/function.hpp>
#include <boost/mp11/bind.hpp>
#include <boost/mp11/utility.hpp>
#include <meta_json_parser/config.h>
#include <meta_json_parser/meta_math.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/parse.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/kernel_launch_configuration.cuh>
#include <type_traits>

namespace JDictOpts {
	struct ConstOrder {};
}

namespace __JDict_internal {
    template<class JDictT, class = void>
    struct InvokeDispatch
    {
        template<class KernelContextT>
        static __device__ __forceinline__ ParsingError dispatch(KernelContextT& kc)
        {
            return JDictT::invoke_base(kc);
        }
    };

    //Temporary solution due to Visual studio compile issues
    template<class JDictT>
    struct InvokeDispatch<JDictT, boost::mp11::mp_list<JDictOpts::ConstOrder>>
    {
        template<class KernelContextT>
        static __device__ __forceinline__ ParsingError dispatch(KernelContextT& kc)
        {
            return JDictT::invoke_const_order(kc);
        }
    };
}

template<class EntriesList, class OptionsT = boost::mp11::mp_list<>>
struct JDict
{
    using type = JDict<EntriesList, OptionsT>;
	using Options = OptionsT;
	using Children = boost::mp11::mp_transform<
		boost::mp11::mp_second,
		EntriesList
	>;
	using Keys = boost::mp11::mp_transform<
		boost::mp11::mp_first,
		EntriesList
	>;
	using UniqueKeys = boost::mp11::mp_equal<
		boost::mp11::mp_size<Keys>,
		boost::mp11::mp_size<boost::mp11::mp_unique<Keys>>
	>;
	static_assert(UniqueKeys::value, "Keys must be unique in JDict.");
	static_assert(boost::mp11::mp_size<JDict>::value, "JDict needs at least 1 entry.");

    template<class JDictT, class>
    friend struct __JDict_internal::InvokeDispatch;

	struct KeyWriter
	{
		using KeyCount = boost::mp11::mp_size<Keys>;
		using Longest = boost::mp11::mp_max_element<
			boost::mp11::mp_transform<
				boost::mp11::mp_size,
				Keys
			>,
			boost::mp11::mp_less
		>;
		using RowCount = boost::mp11::mp_int<(KeyCount::value + 3) / 4>;
		using ColCount = boost::mp11::mp_int<Longest::value * 4>;
		using StorageSize = boost::mp11::mp_int<RowCount::value * ColCount::value>;
		using Buffer = StaticBuffer_c<StorageSize::value>;

		static void __host__ Fill(Buffer& buffer, const KernelLaunchConfiguration* _)
		{
			boost::mp11::mp_for_each<boost::mp11::mp_iota<RowCount>>([&](auto row) {
				boost::mp11::mp_for_each<boost::mp11::mp_iota<ColCount>>([&](auto col) {
					constexpr int KEY_COUNT = KeyCount::value;
					constexpr int COL_COUNT = ColCount::value;
					constexpr int ROW = decltype(row)::value;
					constexpr int COL = decltype(col)::value;
					constexpr int CHAR_ID = COL / 4;
					constexpr int KEY_ID = (ROW * 4) + (3 - (COL % 4));
					using Key = boost::mp11::mp_eval_if_c<
						KEY_ID >= KEY_COUNT,
						boost::mp11::mp_list<>,
						boost::mp11::mp_at,
						Keys,
						boost::mp11::mp_int<KEY_ID>
					>;
					constexpr int KEY_LEN = boost::mp11::mp_size<Key>::value;
					using Char = boost::mp11::mp_eval_if_c <
						CHAR_ID >= KEY_LEN,
						boost::mp11::mp_int<'\0'>,
						boost::mp11::mp_at,
						Key,
						boost::mp11::mp_int<CHAR_ID>
					>;
					buffer.template Alias<char[StorageSize::value]>()
						[ROW * COL_COUNT + COL] = static_cast<char>(Char::value);
				});
			});
		}
	};

	using KeyRequest = FilledMemoryRequest<typename KeyWriter::StorageSize, KeyWriter, MemoryUsage::ReadOnly, MemoryType::Shared>;

	using InternalMemoryRequests = boost::mp11::mp_list<
		KeyRequest,
		ScanRequest<uint32_t>,
		ReduceRequest<uint32_t>
	>;

    using ExternalMemoryRequests = Parse::FindNoneWhiteRequests;

    using MemoryRequests = boost::mp11::mp_append<
        InternalMemoryRequests,
        ExternalMemoryRequests
    >;

private:
	template<class KernelContextT>
	static __device__ __forceinline__ ParsingError invoke_const_order(KernelContextT& kc)
	{
		using KC = KernelContextT;
		using RT = typename KC::RT;
		using WS = typename RT::WorkGroupSize;
		ParsingError err = ParsingError::None;
		if (kc.wgr.PeekChar(0) != '{')
			return ParsingError::Other;
		kc.wgr.AdvanceBy(1);
		char c;
		err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
		if (err != ParsingError::None)
			return err;
		boost::mp11::mp_for_each<Keys>([&](auto key) {
			if (err != ParsingError::None)
				return;
			using Key = decltype(key);
			using KeyPos = boost::mp11::mp_find<Keys, Key>;
			typename KeyWriter::Buffer& keys = kc.m3
				.template Receive<KeyRequest>()
				.template Alias<typename KeyWriter::Buffer>();
			constexpr int OUTER_LOOPS = (KeyWriter::KeyCount::value + 31) / 32;
			constexpr uint32_t keyMask = (0xFFu << (3 - (KeyPos::value % 4)) * 8);
			static_assert(OUTER_LOOPS == 1, "JDict currently is hardcoded for max 32 keys.");
			boost::mp11::mp_for_each<boost::mp11::mp_iota_c<OUTER_LOOPS>>([&](auto i) {
				if (err != ParsingError::None)
					return;
				int stringReadIdx = 0;
				bool keyMatch = true;
				err = JsonParse::String(kc, [&](bool& isEscaped, int& activeThreads) {
					char c = kc.wgr.CurrentChar();
					char key = '\0';
					if (stringReadIdx * RT::GroupSize() + RT::WorkerId() < KeyWriter::Longest::value)
					{
						//Each row contains 4 combined keys.
						uint32_t key4 = reinterpret_cast<uint32_t*>(&keys)[
							KeyWriter::Longest::value * (KeyPos::value / 4) + //Key offset
							stringReadIdx * RT::GroupSize() + //In-key offset
							RT::WorkerId() //Thread offset
						];
						key4 &= keyMask;
						key4 >>= (3 - (KeyPos::value % 4)) * 8;
						key = static_cast<char>(key4);
					}

					keyMatch &= key == (RT::WorkerId() < activeThreads ? c : '\0');
					keyMatch = kc.wgr.all_sync(keyMatch);
					++stringReadIdx;
					return keyMatch ? ParsingError::None : ParsingError::Other;
				});
                __syncwarp();
				if (err != ParsingError::None)
					return;
				err = Parse::FindNext<':', WS>::KC(kc).template Do<Parse::StopTag::StopAfter>();
				if (err != ParsingError::None)
					return;
				err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
				if (err != ParsingError::None)
					return;
				using Action = boost::mp11::mp_second<boost::mp11::mp_map_find<
					EntriesList,
					Key
					>>;
				err = Action::Invoke(kc);
				return;
			});
			if (err != ParsingError::None)
				return;
			err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
			if (err != ParsingError::None)
				return;
			c = kc.wgr.PeekChar(0);
			kc.wgr.AdvanceBy(1);
			if (c != boost::mp11::mp_if_c <
				KeyPos::value < boost::mp11::mp_size<Keys>::value - 1,
				std::integral_constant<char, ','>,
				std::integral_constant<char, '}'>
				>::value)
			{
				err = ParsingError::Other;
				return;
			}
			if (KeyPos::value < boost::mp11::mp_size<Keys>::value - 1)
			{
				err = Parse::FindNext<'"', WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
				if (err != ParsingError::None)
					return;
			}
		});
		return err;
	}

	template<class KernelContextT>
	static __device__ __forceinline__ ParsingError invoke_base(KernelContextT& kc)
	{
		using KC = KernelContextT;
		using RT = typename KC::RT;
		using WS = typename RT::WorkGroupSize;
		ParsingError err;
		if (kc.wgr.PeekChar(0) != '{')
			return ParsingError::Other;
		kc.wgr.AdvanceBy(1);
		char c;
		err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
		if (err != ParsingError::None)
			return err;
		c = kc.wgr.PeekChar(0);
		if (c == '}')
		{
			kc.wgr.AdvanceBy(1);
			return ParsingError::None;
		}
		do
		{
			typename KeyWriter::Buffer& keys = kc.m3
				.template Receive<KeyRequest>()
				.template Alias<typename KeyWriter::Buffer>();
			constexpr int OUTER_LOOPS = (KeyWriter::KeyCount::value + 31) / 32;
			static_assert(OUTER_LOOPS == 1, "JDict currently is hardcoded for max 32 keys.");
			boost::mp11::mp_for_each<boost::mp11::mp_iota_c<OUTER_LOOPS>>([&](auto i) {
				if (err != ParsingError::None)
					return;
				constexpr int OUTER_I = decltype(i)::value;
				//keyBits = i-th bit set indicates that i-th key was not disqualified    
				//at the beginning bit is set for each key.
				//e.g. if we are checking first 32 keys of out 42 keys, mask will be 0xFF'FF'FF'FF
				//if we are checking only 10 remaining keys, mask will be 0x00'00'03'FF
				uint32_t keyBits = OUTER_I == (OUTER_LOOPS - 1)
					? 0xFF'FF'FF'FFu ^ ((0x1u << (32 - KeyWriter::KeyCount::value)) - 1)
					: 0x0u;
				int stringReadIdx = 0;
				err = JsonParse::String(kc, [&](bool& isEscaped, int& activeThreads) {
					//Each row contains 4 combined keys.
					constexpr int ROW_COUNT = KeyWriter::RowCount::value;
					char c = kc.wgr.CurrentChar();
					char4 c4{ c, c, c, c };
					boost::mp11::mp_for_each<boost::mp11::mp_iota_c<ROW_COUNT>>([&](auto j) {
						constexpr int INNER_I = decltype(j)::value;
						if (INNER_I != 0 && INNER_I % 32 == 0)
						{
							static_assert(KeyWriter::KeyCount::value <= 32, "JDict currently is hardcoded for max 32 keys.");
						}
						//Example (. for null bytes):
						//INNER_I = 2
						//4key = . a 1 x
						//c = 'a'
						//keyBytes = 0x00'FF'00'00
						//_ & 0x01'01'01'01 -> 0x00'01'00'00
						//(_ >> 7) | _ -> 0x__'01'__'00
						//(_ >> 14) | _ -> 0x__'__'__'04
						//_ & 0x0F -> 0x4
						//_ | 0xFF'FF'FF'F0 -> 0xFF'FF'FF'F4
						//_:0xFF'FF'FF'FF << (28 - 4 * 2) -> 0xFF'4F'FF'FF
						char4 key4{ '\0', '\0', '\0', '\0' };
						if (stringReadIdx * RT::GroupSize() + RT::WorkerId() < KeyWriter::Longest::value)
							key4 = reinterpret_cast<char4*>(&keys)[
								KeyWriter::Longest::value * INNER_I + //Key offset
								stringReadIdx * RT::GroupSize() + //In-key offset
								RT::WorkerId() //Thread offset
							];
						//If thread is inactive, it should vote for match of a key only if key has ended.
						// + Set bits for inactive threads
						uint32_t keyByte = threadIdx.x < activeThreads ? 0x00'00'00'00u : 0xFF'FF'FF'FFu;
						// + Keep bits only if there is no key chars to check
						keyByte &= __vcmpeq4(BytesCast<uint32_t>(key4), 0x00'00'00'00u);
						// Set bits if key chars match an input
						keyByte |= __vcmpeq4(BytesCast<uint32_t>(key4), BytesCast<uint32_t>(c4));
						keyByte &= 0x01'01'01'01u;
						keyByte |= keyByte >> 7;
						keyByte |= keyByte >> 14;
						keyByte &= 0x0F;
						constexpr int shift_by = (32 - 4) - ((INNER_I % 32) * 4);
						keyByte = __funnelshift_lc(0xFF'FF'FF'FFu, keyByte | 0xFF'FF'FF'F0u, shift_by);
						keyBits &= keyByte;
					});
					++stringReadIdx;
					return ParsingError::None;
				});
				if (err != ParsingError::None)
					return;
				keyBits = Reduce<uint32_t, WS>(kc).Reduce(keyBits, BitAnd());
				keyBits = Scan<uint32_t, WS>(kc).Broadcast(keyBits, 0);
				err = Parse::FindNext<':', WS>::KC(kc).template Do<Parse::StopTag::StopAfter>();
				if (err != ParsingError::None)
					return;
				err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
				if (err != ParsingError::None)
					return;
				if (keyBits == 0)
				{
					//TODO skip
					assert(false);
					err = ParsingError::Other;
					return;
					//err = SKIP
					//if (err != ParsingError::None)
					//	return err;
				}
				else
				{
					boost::mp11::mp_for_each<boost::mp11::mp_iota_c<KeyWriter::KeyCount::value>>([&](auto key_index) {
						if (err != ParsingError::None)
							return;
						constexpr int KEY_INDEX = decltype(key_index)::value;
						using Key = boost::mp11::mp_at_c<Keys, KEY_INDEX>;
						if (keyBits & (0x80'00'00'00u >> KEY_INDEX))
						{
							using Action = boost::mp11::mp_second<boost::mp11::mp_map_find<
								EntriesList,
								Key
							>>;
							err = Action::Invoke(kc);
						}
					});
				}
				return;
			});
			if (err != ParsingError::None)
				return err;
			err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
			if (err != ParsingError::None)
				return err;
			c = kc.wgr.PeekChar(0);
			kc.wgr.AdvanceBy(1);
			if (c == '}')
				break;
			if (c != ',')
				return ParsingError::Other;
			err = Parse::FindNext<'"', WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
			if (err != ParsingError::None)
				return err;
		} while (true);
		return ParsingError::None;
	}

public:
	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{

        return __JDict_internal::InvokeDispatch<type, Options>::dispatch(kc);
	}
};
