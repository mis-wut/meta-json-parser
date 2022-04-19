// corresponding JSON file: float.json
// { "generic": -1e6, "scientific": -1.2345e2, "fixed": -12.2345 }

#include <meta_json_parser/meta_utility/metastring.h>

// KEYS
using K_L1_generic = metastring("generic");
using K_L1_scientific = metastring("scientific");
using K_L1_fixed = metastring("fixed");

// OPTIONS
using JRealOptionsFixedFormat = boost::mp11::mp_list<
    boost::mp11::mp_list<
        JRealNumberOptions::JRealNumberExponent,
        JRealNumberOptions::JRealNumberExponent::WithoutExponent
    >
>;

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
    mp_list<K_L1_generic, JRealNumber<float, K_L1_generic>>,
	mp_list<K_L1_scientific, JRealNumber<double, K_L1_scientific>>,
	mp_list<K_L1_fixed, JRealNumber<float, K_L1_fixed, JRealOptionsFixedFormat>>,
>,
    DictOpts
> ;
