#pragma once
#include <boost/mp11/list.hpp>
#include <boost/mp11/bind.hpp>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/action/void_action.cuh>

template<class T, typename = int>
struct HaveMemoryRequests : std::false_type {};

template<class T>
struct HaveMemoryRequests<T, decltype(std::declval<typename T::MemoryRequests>(), 0)> : std::true_type {};

template<class T>
using GetMemoryRequests = typename T::MemoryRequests;

template<class T>
using TryGetMemoryRequests = boost::mp11::mp_eval_if_not<
	HaveMemoryRequests<T>,
	boost::mp11::mp_list<>,
	GetMemoryRequests,
	T
>;

using GetMemoryRequests_q = boost::mp11::mp_quote<GetMemoryRequests>;

template<class BaseActionT, class AdditionalRequestsT = boost::mp11::mp_list<>>
struct MemoryConfiguration
{
	using BaseAction = BaseActionT;
	using AdditionalRequests = AdditionalRequestsT;
	using ActionRequests = boost::mp11::mp_flatten<
			boost::mp11::mp_transform<
			TryGetMemoryRequests,
			ActionIterator<BaseActionT>
		>
	>;

	using AllRequests = boost::mp11::mp_append<
		ActionRequests,
		AdditionalRequests
	>;

	using ReadOnlyList = boost::mp11::mp_copy_if_q<
		AllRequests,
		boost::mp11::mp_compose_q<
			boost::mp11::mp_quote<GetRequestMemoryUsage>,
			boost::mp11::mp_bind<
				boost::mp11::mp_same,
				boost::mp11::_1,
				MemoryUsage::ReadOnly
			>
		>
	>;
	using ActionList = boost::mp11::mp_copy_if_q<
		AllRequests,
		boost::mp11::mp_compose_q<
			boost::mp11::mp_quote<GetRequestMemoryUsage>,
			boost::mp11::mp_bind<
				boost::mp11::mp_same,
				boost::mp11::_1,
				MemoryUsage::ActionUsage
			>
		>
	>;
	using AtomicList = boost::mp11::mp_copy_if_q<
		AllRequests,
		boost::mp11::mp_compose_q<
			boost::mp11::mp_quote<GetRequestMemoryUsage>,
			boost::mp11::mp_bind<
				boost::mp11::mp_same,
				boost::mp11::_1,
				MemoryUsage::AtomicUsage
			>
		>
	>;
};

using EmptyMemoryConfiguration = MemoryConfiguration<VoidAction>;

