#pragma once
#include <type_traits>
#include <boost/mp11/list.hpp>

//template<class TypeT, template<typename...> typename TemplateT>
//using IsTemplate = std::is_same<
//	TypeT,
//	boost::mp11::mp_rename<
//		TypeT,
//		TemplateT
//	>
//>;

template<template<typename...> typename TemplateT>
struct IsTemplate
{
	template<class TypeT>
	using fn = std::is_same<
		TypeT,
		boost::mp11::mp_rename<
			TypeT,
			TemplateT
		>
	>;
};

//template<class TypeT, class TemplateQ>
//using IsTemplate_q = IsTemplate<TypeT, TemplateQ::template fn>;

