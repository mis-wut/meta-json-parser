#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/function.hpp>
#include <boost/mp11/bind.hpp>
#include <meta_json_parser/config.h>
#include <meta_json_parser/meta_math.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/parse.cuh>
#include <type_traits>

template<class EntriesList>
struct JArray
{
	using Indices = boost::mp11::mp_transform<
		boost::mp11::mp_first,
		EntriesList
	>;
	using UniqueIndices = boost::mp11::mp_equal<
		boost::mp11::mp_size<Indices>,
		boost::mp11::mp_size<boost::mp11::mp_unique<Indices>>
	>;
	static_assert(UniqueIndices::value, "Indices must be unique in JArray.");
	static_assert(boost::mp11::mp_size<EntriesList>::value, "JArray needs at least 1 entry.");
	using SortedEntries = boost::mp11::mp_sort_q<
		EntriesList,
		boost::mp11::mp_bind_q<
			boost::mp11::mp_quote<boost::mp11::mp_less>,
			boost::mp11::mp_bind<boost::mp11::mp_first, boost::mp11::_1>,
			boost::mp11::mp_bind<boost::mp11::mp_first, boost::mp11::_2>
		>
	>;
	using SortedIndices = boost::mp11::mp_transform<
		boost::mp11::mp_first,
		SortedEntries
	>;
	using MaxIndex = boost::mp11::mp_back<SortedIndices>;

	template<class Idx, class KernelContextT>
	static __device__ INLINE_METHOD typename std::enable_if<
		std::is_same_v<
			boost::mp11::mp_map_find<
				SortedEntries,
				Idx
			>,
			void
		>,
		ParsingError
	>::type DispatchIndex(KernelContextT& kc)
	{
		//TODO add skip
		assert(0);
		//return SKIP
		return ParsingError::None;
	}

	template<class Idx, class KernelContextT>
	static __device__ INLINE_METHOD typename std::enable_if<
		!std::is_same_v<
			boost::mp11::mp_map_find<
				SortedEntries,
				Idx
			>,
			void
		>,
		ParsingError
	>::type DispatchIndex(KernelContextT& kc)
	{
		using Action = boost::mp11::mp_map_find<
			SortedEntries,
			Idx
		>;
		return Action::Invoke(kc);
	}

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		using KC = KernelContextT;
		using RT = typename KC::RT;
		using WS = typename RT::WorkGroupSize;
		if (kc.wgr.PeekChar(0) != '[')
			return ParsingError::Other;
		kc.wgr.AdvanceBy(1);
		ParsingError err = ParsingError::None;
		err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
		if (err != ParsingError::None)
			return err;
		char c = kc.wgr.PeekChar(0);
		if (c == ']')
		{
			kc.wgr.AdvanceBy(1);
			return ParsingError::None;
		}
		bool endOfArray = false;
		boost::mp11::mp_for_each<boost::mp11::mp_iota<MaxIndex>>([&](auto i)
		{
			using Idx = decltype(i);
			if (err != ParsingError::None || endOfArray)
				return;
			err = DispatchIndex<Idx>(kc);
			if (err != ParsingError::None)
				return;
			err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
			if (err != ParsingError::None)
				return;
			c = kc.wgr.PeekChar(0);
			kc.wgr.AdvanceBy(1);
			if (c == ',')
			{
				err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
				if (err != ParsingError::None)
					return;
			}
			else if (c == ']')
			{
				endOfArray = true;
				return;
			}
			else
			{
				err = ParsingError::Other;
				return;
			}
		});
		if (err != ParsingError::None)
			return err;
		if (endOfArray)
			return ParsingError::None;
		while (true)
		{
			//TODO add skip
			//err = SKIP
			assert(0);
			if (err != ParsingError::None)
				return err;
			err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
			if (err != ParsingError::None)
				return err;
			c = kc.wgr.PeekChar(0);
			if (c == ',')
			{
				err = Parse::FindNoneWhite<WS>::KC(kc).template Do<Parse::StopTag::StopAt>();
				if (err != ParsingError::None)
					return err;
			}
			else if (c == ']')
				break;
			else
				return ParsingError::Other;
		}
		return ParsingError::None;
	}
};