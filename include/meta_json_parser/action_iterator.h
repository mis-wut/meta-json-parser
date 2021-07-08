#pragma once
#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/bind.hpp>
#include <type_traits>

template<class T, typename = int>
struct HaveChildren : std::false_type {};

template<class T>
struct HaveChildren<T, decltype(std::declval<typename T::Children>(), 0)> : std::true_type {};

template<class BaseAction>
using GetChildren = typename BaseAction::Children;

using GetChildren_q = boost::mp11::mp_quote<GetChildren>;

template<class BaseAction, class = void>
struct ActionIterator_impl
{
	using type = boost::mp11::mp_list<BaseAction>;
};

template<class BaseAction>
struct ActionIterator_impl<
	BaseAction,
	typename std::enable_if<
		HaveChildren<BaseAction>::value
	>::type
>
{
	using children = GetChildren<BaseAction>;
	using transformed_children =
		boost::mp11::mp_transform_q<
			boost::mp11::mp_quote_trait<ActionIterator_impl>,
			children
		>;
	using flatten_children = boost::mp11::mp_flatten<transformed_children>;
	using type = boost::mp11::mp_push_front<flatten_children, BaseAction>;
};

template<class BaseAction>
using ActionIterator = typename ActionIterator_impl<BaseAction>::type;

using ActionIterator_q = boost::mp11::mp_quote_trait<ActionIterator_impl>;
