#pragma once
#include <type_traits>

template<class T, typename = int>
struct HaveParserRequirements : std::false_type {};

template<class T>
struct HaveParserRequirements<T, decltype(std::declval<typename T::ParserRequirements>(), 0)> : std::true_type {};

template<class T>
using GetParserRequirements = typename T::ParserRequirements;

namespace ParserRequirement
{
	struct KeepDistance {};
}