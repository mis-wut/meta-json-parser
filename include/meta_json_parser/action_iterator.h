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

namespace impl_tagged_action {
	template<class T, typename = int>
	struct HaveTag : std::false_type {};

	template<class T>
	struct HaveTag<T, decltype(std::declval<typename T::Tag>(), 0)> : std::true_type {};

	template<class T>
	using GetTag = typename T::Tag;

    template<class BaseActionT, class TagT>
    struct impl_GetTaggedAction {
        using Tag = TagT;
        template<class T>
        using SameTag = boost::mp11::mp_same<Tag, T>;

        template<class Action>
        using HaveSameTag = boost::mp11::mp_eval_if_not_q<
            HaveTag<Action>,
            boost::mp11::mp_false,
            boost::mp11::mp_compose<
                impl_tagged_action::GetTag,
                SameTag
            >,
            Action
        >;

        using Actions = ActionIterator<BaseActionT>;

        using WithSameTag = boost::mp11::mp_copy_if<Actions, HaveSameTag>;

        using type = boost::mp11::mp_first<WithSameTag>;
    };

}

//find(action_list, (action) -> {
//	if HaveTag(action)
//		return false
//	return same(TagT, GetTag(action))
//})
template<class BaseActionT, class TagT>
using GetTaggedAction = typename impl_tagged_action::impl_GetTaggedAction<BaseActionT, TagT>::type;

		

