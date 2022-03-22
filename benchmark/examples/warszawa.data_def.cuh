// INCLUDES
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jrealnumber.cuh>
#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/meta_utility/metastring.h>

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
using JRealOptionsFixedFormat = boost::mp11::mp_list<
    boost::mp11::mp_list< // key: value
        JRealNumberOptions::JRealNumberExponent,
        JRealNumberOptions::JRealNumberExponent::WithoutExponent
    >
>;

using JDatetimeResolutionSeconds = boost::mp11::mp_list<
  boost::mp11::mp_list<
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
	mp_list<K_L1_Time, JDatetime<DatetimeFormat_YYYYMMDD_HHMMSS, uint64_t, K_L1_Time, JDatetimeResolutionSeconds>>,
	mp_list<K_L1_Lat, JRealNumber<float, K_L1_Lat, JRealOptionsFixedFormat>>,
	mp_list<K_L1_Brigade, StringFun<K_L1_Brigade, STATIC_STRING_SIZE>>
>,
    DictOpts
> ;
