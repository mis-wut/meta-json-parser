#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <boost/mp11.hpp>

#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/meta_utility/metastring.h>

#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jrealnumber.cuh>
#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/action/jbool.cuh>

#include <meta_json_parser/action/jstring_custom.cuh>
#include <meta_json_parser/action/string_transform_functors/polynomial_rolling_hash_matcher.cuh>

using namespace boost::mp11;
using namespace std;

// SETTINGS
using WorkGroupSize = mp_int<32>;

// EXAMPLE:
/*
  {
    "date": "2010.01.01",
    "result": "1-0",
    "phone": "856 249 140",
    "status": "BETA",
    "score_1-5": 2,
    "lat": -20.07,
    "lon": 89.17,
    "lognormal": 322.1996679417,
    "poisson": 9,
    "pts": 73.808461031,
    "is_checked": false,
    "randint": 2989,
    "name_NNN": "name_526"
  } 
 */

// KEYS (Key, Level 1, ...)
using K_L1_date = metastring("date");
using K_L1_result = metastring("result");
using K_L1_phone  = metastring("phone");
using K_L1_status = metastring("status");
using K_L1_score = metastring("score_1-5");
using K_L1_lat = metastring("lat");
using K_L1_lon = metastring("lon");
using K_L1_lognormal = metastring("lognormal");
using K_L1_poisson = metastring("poisson");
using K_L1_pts = metastring("pts");
using K_L1_is_checked = metastring("is_checked");
using K_L1_randint = metastring("randint");
using K_L1_name_NNN = metastring("name_NNN");

// FORMATS
using DatetimeFormat_YMD = metastring("%Y.%m.%d");

// OPTIONS
using JRealOptionsFixedFormat = mp_list<
    mp_list< // key: value
        JRealNumberOptions::JRealNumberExponent,
        JRealNumberOptions::JRealNumberExponent::WithoutExponent
    >
>;

// TODO: support JDatetimeResultionDays, when it becomes available
using JDatetimeResolutionSeconds = mp_list<
  mp_list<
    JDatetimeOptions::TimestampResolution,
    JDatetimeOptions::TimestampResolution::Seconds
  >
>;



template<uint64_t N>
using u64 = std::integral_constant<uint64_t, N>;
using StringMap = boost::mp11::mp_list<
    boost::mp11::mp_list<
        boost::mp11::mp_string<'1', '-', '0'>,
        u64<1>
    >,
    boost::mp11::mp_list<
        boost::mp11::mp_string<'0', '-', '1'>,
        u64<2>
    >,
    boost::mp11::mp_list<
        boost::mp11::mp_string<'1', '/', '2', '-', '1', '/','2'>,
        u64<3>
    >,
    boost::mp11::mp_list<
        boost::mp11::mp_string<'*'>,
        u64<4>
    >
>;

using Multiplier = u64<31>;
using Modulus = u64<static_cast<uint64_t>(1e9 + 9)>;
using Tag = int64_t;
using Functor = PolynomialRollingHashMatcher<Multiplier, Modulus, StringMap, Tag>;
using Action = JStringCustom<Functor>;

// CONFIGURE USE OF EXTRA COLUMN TYPES
#define USE_CATEGORICAL
#define USE_DATETIME

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
#ifdef USE_DATETIME
    mp_list<K_L1_date, JDatetime<DatetimeFormat_YMD, int64_t, K_L1_date, JDatetimeResolutionSeconds>>,
#else
#pragma message("Not using JDatetime for parsing, but JString / StringFun")
    mp_list<K_L1_date, StringFun<K_L1_date, STATIC_STRING_SIZE>>,
#endif
#ifdef USE_CATEGORICAL
    mp_list<K_L1_result, Action>, // TODO: categorical
#else
#pragma message("Not using JStringCustom for parsing categorical data")
    mp_list<K_L1_result, StringFun<K_L1_result, STATIC_STRING_SIZE>>,
#endif
    mp_list<K_L1_phone, StringFun<K_L1_phone, STATIC_STRING_SIZE>>, // TODO: transformation
    mp_list<K_L1_status, StringFun<K_L1_status, STATIC_STRING_SIZE>>, // TODO: transformation
    mp_list<K_L1_score, JNumber<uint32_t, K_L1_score>>, // TODO: transform into float \in 0..1
    mp_list<K_L1_lat, JRealNumber<float, K_L1_lat, JRealOptionsFixedFormat>>,
    mp_list<K_L1_lon, JRealNumber<float, K_L1_lon, JRealOptionsFixedFormat>>,
    mp_list<K_L1_lognormal, JRealNumber<double, K_L1_lognormal>>, // TODO: cut (???)
    mp_list<K_L1_poisson, JNumber<uint64_t, K_L1_poisson>>, // TODO: cut (???)
    mp_list<K_L1_pts, JRealNumber<float, K_L1_pts>>, // TODO: cut
    mp_list<K_L1_is_checked, JBool<uint8_t, K_L1_is_checked>>, // NOTE: must be uint8_t
    mp_list<K_L1_randint, JNumber<uint16_t, K_L1_randint>>,
    mp_list<K_L1_name_NNN, StringFun<K_L1_name_NNN, STATIC_STRING_SIZE>>
>,
    DictOpts
> ;

// NOTE: Neither PARSER OPTIONS nor PARSER are needed for 'data_def.cuh'
// that is for inclusion in the 'benchmark/main.cu'

// PARSER OPTIONS
//template<class Key, int Size>
//using StaticCopyFun = JStringStaticCopy<mp_int<Size>, Key>;

// PARSER
//using BaseAction = DictCreator<StaticCopyFun, mp_list<>>;

#endif /* !defined(META_CUDF_META_DEF_CUH) */
