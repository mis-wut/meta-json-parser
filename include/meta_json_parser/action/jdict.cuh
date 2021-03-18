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
#include <type_traits>

template<class EntriesList>
struct JDict
{
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

	template<class T>
	using GetOutputRequests = boost::mp11::mp_second<T>::OutputRequests;

	template<class T>
	using GetMemoryRequests = boost::mp11::mp_second<T>::MemoryRequests;

	using OutputRequests = boost::mp11::mp_flatten<boost::mp11::mp_transform<
		GetOutputRequests,
		EntriesList
	>>;

	using MemoryRequests = boost::mp11::mp_flatten<boost::mp11::mp_transform<
		GetMemoryRequests,
		EntriesList
	>>;

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

		static void __host__ Fill(Buffer& buffer)
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
};
