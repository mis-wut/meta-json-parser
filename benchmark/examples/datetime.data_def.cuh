// INCLUDES
#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// { "date": "2021-03-18 00:16:48" }

// KEYS
using K_L1_date = mp_string<'d', 'a', 't', 'e'>;

// FORMATS
using DatetimeFormat_YYYYMMDD_HHMMSS = metastring("%Y-%m-%d %H:%M:%S");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
    mp_list<K_L1_date, JDatetime<DatetimeFormat_YYYYMMDD_HHMMSS, int64_t, K_L1_date>>,
>,
    DictOpts
> ;
