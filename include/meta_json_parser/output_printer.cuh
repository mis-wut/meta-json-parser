#pragma once
#include <iostream>
#include <type_traits>
#include <boost/mp11/map.hpp>
#include <meta_json_parser/parser_output_host.cuh>
#include <meta_json_parser/output_manager.cuh>

template<class, class = void>
struct support_ostream : std::false_type {};

template<class T>
struct support_ostream<T, std::void_t<decltype(std::declval<std::ostream&>() <<
                                         std::declval<T>())>> : std::true_type {};

template<class T, class = typename std::enable_if<support_ostream<T>::value>::type>
struct DefaultPrinter
{
	void operator()(std::ostream& stream, const T* ptr) const
	{
		stream << *ptr;
	}
};

template<class T>
struct AsCharsPrinter
{
	void operator()(std::ostream& stream, const T* ptr) const
	{
		const char* c_str = reinterpret_cast<const char*>(ptr);
		if (c_str[sizeof(T) - 1] == '\0')
			stream << c_str;
		else
			stream.write(c_str, sizeof(T));
	}
};

template<class T>
struct BoolPrinter
{
	void operator()(std::ostream& stream, const T* ptr) const
	{
		stream << (*ptr ? "true" : "false");
	}
};

template<class BaseActionT, class PrinterMapT>
struct OutputPrinter
{
	using OC = OutputConfiguration<typename BaseActionT::OutputRequests>;
	using OM = OutputManager<OC>;

	template<class OutputTagT>
	using GetOutType = typename boost::mp11::mp_map_find<
		typename OC::RequestList,
		OutputTagT
	>::OutputType;

	template<class OutputTagT>
	using MapFound = boost::mp11::mp_map_find<PrinterMapT, OutputTagT>;

	template<class OutputTagT>
	using InMap = boost::mp11::mp_not<std::is_same<
		void,
		MapFound<OutputTagT>
	>>;
	
	template<class OutputTagT>
	using EvaluatedPrinter = typename boost::mp11::mp_second<MapFound<OutputTagT>>
		::template fn <GetOutType<OutputTagT>>;

	template<class OutputTagT>
	using GetPrinter = boost::mp11::mp_eval_if_not<
		InMap<OutputTagT>,
		boost::mp11::mp_eval_or<
			AsCharsPrinter<GetOutType<OutputTagT>>,
			DefaultPrinter,
			GetOutType<OutputTagT>
		>,
		EvaluatedPrinter,
		OutputTagT
	>;

	void ToCsv(std::ostream& stream, ParserOutputHost<BaseActionT>& host_output)
	{
		for (auto i = 0ull; i < host_output.m_size; ++i)
		{
			boost::mp11::mp_for_each<typename OC::RequestList>([&, idx=0](auto k) mutable {
				using Request = decltype(k);
				using Tag = typename Request::OutputTag;
				using T = typename Request::OutputType;
				if (idx != 0)
					stream << ',';
				const uint8_t* ptr = host_output.m_h_outputs[idx++].data();
				const T* cast_ptr = reinterpret_cast<const T*>(ptr);
				GetPrinter<Tag> printer;
				printer(stream, cast_ptr);
			});
			stream << '\n';
		}
	}
};
