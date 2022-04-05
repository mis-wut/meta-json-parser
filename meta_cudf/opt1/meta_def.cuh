#ifndef MEDA_CUDF_META_DEF_CUH
#define MEDA_CUDF_META_DEF_CUH


// INCLUDES
#include <boost/mp11.hpp>

/* #include <meta_json_parser/config.h> */

/* #include <meta_json_parser/parser_output_device.cuh> */
/* #include <meta_json_parser/output_printer.cuh> */
/* #include <meta_json_parser/memory_configuration.h> */
/* #include <meta_json_parser/runtime_configuration.cuh> */
/* #include <meta_json_parser/parser_configuration.h> */

/* #include <meta_json_parser/parsing_error.h> */
/* #include <meta_json_parser/parser_kernel.cuh> */
/* #include <meta_json_parser/mp_string.h> */

#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jrealnumber.cuh>
#include <meta_json_parser/action/datetime/jdatetime.cuh>
/* #include <meta_json_parser/action/jbool.cuh> */

#include <meta_json_parser/meta_utility/metastring.h>

using namespace boost::mp11;
using namespace std;


template<class Key, int Size>
using StaticCopyFun = JStringStaticCopy<mp_int<Size>, Key>;

using WorkGroupSize = mp_int<32>;
/* using BaseAction = JNumber<int, void>; */

// EXAMPLE:
// {"Lines":"192","Lon":20.9953361,"VehicleNumber":"1026","Time":"2021-03-18 00:16:48","Lat":52.1881257,"Brigade":"3"}

// KEYS
using K_L1_Lines = metastring("Lines");
using K_L1_Lon = metastring("Lon");
using K_L1_Lat = metastring("Lat");
using K_L1_Vehicle = metastring("VehicleNumber");
using K_L1_Brigade = metastring("Brigade");
using K_L1_Time = metastring("Time");

// FORMATS
using DatetimeFormat_YYYYMMDD_HHMMSS = metastring("%Y-%m-%d %H:%M:%S");

// OPTIONS
using JRealOptionsFixedFormat = mp_list<
    mp_list< // key: value
        JRealNumberOptions::JRealNumberExponent,
        JRealNumberOptions::JRealNumberExponent::WithoutExponent
    >
>;

using JDatetimeResolutionSeconds = mp_list<
  mp_list<
    JDatetimeOptions::TimestampResolution,
    JDatetimeOptions::TimestampResolution::Seconds
  >
>;

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
	mp_list<K_L1_Lines, StringFun<K_L1_Lines, STATIC_STRING_SIZE>>,
	mp_list<K_L1_Lon, JRealNumber<float, K_L1_Lon, JRealOptionsFixedFormat>>,
	mp_list<K_L1_Vehicle, StringFun<K_L1_Vehicle, STATIC_STRING_SIZE>>,
	mp_list<K_L1_Time, StringFun<K_L1_Time, STATIC_STRING_SIZE>>,
	mp_list<K_L1_Lat, JRealNumber<float, K_L1_Lat, JRealOptionsFixedFormat>>,
	mp_list<K_L1_Brigade, StringFun<K_L1_Brigade, STATIC_STRING_SIZE>>
>,
    DictOpts
>;

using BaseAction = DictCreator<StaticCopyFun, mp_list<>>;


#endif //META_CUDF_META_DEF_CUH
