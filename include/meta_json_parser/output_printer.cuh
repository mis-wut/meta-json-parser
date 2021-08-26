#pragma once
#include <iostream>
#include <type_traits>
#include <boost/mp11/map.hpp>
#include <meta_json_parser/output_manager.cuh>

template<class, class = void>
struct support_ostream : std::false_type {};

template<class T>
struct support_ostream<T, std::void_t<decltype(std::declval<std::ostream&>() <<
                                         std::declval<T>())>> : std::true_type {};

namespace impl_printer {
	template<class T, typename = int>
	struct HavePrinter : std::false_type {};

	template<class T>
	struct HavePrinter<T, decltype(std::declval<typename T::Printer>(), 0)> : std::true_type {};

	template<class T>
	using GetPrinter = typename T::Printer;

	template<class ActionT>
	using FirstOutputType = typename boost::mp11::mp_first<typename ActionT::OutputRequests>::OutputType;
}

template<class ActionT>
struct DefaultPrinter
{
	using Tag = typename ActionT::Tag;
	using OutputT = impl_printer::FirstOutputType<ActionT>;

	template<class ParserOutputHostT, class = typename std::enable_if<support_ostream<impl_printer::FirstOutputType<ActionT>>::value>::type>
	void static Print(const ParserOutputHostT& output, size_t idx, std::ostream& stream)
	{
		stream << reinterpret_cast<const OutputT*>(output.template Pointer<Tag>())[idx];
	}
};

template<class ActionT>
struct AsCharsPrinter
{
	using Tag = typename ActionT::Tag;
	using OutputType = impl_printer::FirstOutputType<ActionT>;

	template<class ParserOutputHostT>
	void static Print(const ParserOutputHostT& output, size_t idx, std::ostream& stream)
	{
		const char* c_str = reinterpret_cast<const char*>(
			reinterpret_cast<const OutputType*>(output.template Pointer<Tag>()) + idx
		);
		if (c_str[sizeof(OutputType) - 1] == '\0')
			stream << c_str;
		else
			stream.write(c_str, sizeof(OutputType));
	}
};

template<class ActionT>
struct BoolPrinter
{
	using Tag = typename ActionT::Tag;
	using OutputType = impl_printer::FirstOutputType<ActionT>;

	template<class ParserOutputHostT>
	void static Print(const ParserOutputHostT& output, size_t idx, std::ostream& stream)
	{
		stream << (reinterpret_cast<const OutputType*>(output.template Pointer<Tag>())[idx] ? "true" : "false");
	}
};

template<class ActionT>
using GetPrinter = boost::mp11::mp_eval_if_not<
	impl_printer::HavePrinter<ActionT>,
	DefaultPrinter<ActionT>,
	impl_printer::GetPrinter,
	ActionT
>;
