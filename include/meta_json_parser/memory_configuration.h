#pragma once
#include <boost/mp11/list.hpp>
#include <boost/mp11/bind.hpp>
#include <meta_json_parser/memory_request.h>

template<class ReadOnlyListT, class ActionListT, class AtomicListT>
struct MemoryConfiguration
{
	using ReadOnlyList = ReadOnlyListT;
	using ActionList = ActionListT;
	using AtomicList = AtomicListT;
};

template<class MemoryConfigurationT, class MemoryRequestT>
using AppendRequest = boost::mp11::mp_rename<
	boost::mp11::mp_second<
		boost::mp11::mp_map_find<
			boost::mp11::mp_list<
				boost::mp11::mp_list<MemoryUsage::ReadOnly, boost::mp11::mp_quote<boost::mp11::mp_transform_first_q>>,
				boost::mp11::mp_list<MemoryUsage::ActionUsage, boost::mp11::mp_quote<boost::mp11::mp_transform_second_q>>,
				boost::mp11::mp_list<MemoryUsage::AtomicUsage, boost::mp11::mp_quote<boost::mp11::mp_transform_third_q>>
			>,
		typename MemoryRequestT::MemoryUsage
	>>::fn<
		MemoryConfigurationT,
		boost::mp11::mp_bind<
			boost::mp11::mp_push_back,
			boost::mp11::_1,
			MemoryRequestT
		>
	>,
	MemoryConfiguration
>;
