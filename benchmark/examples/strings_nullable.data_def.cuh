// INCLUDES
#include <meta_json_parser/meta_utility/metastring.h>
#include <meta_json_parser/action/decorators/null_default_value.cuh>

// EXAMPLE:
// {"name":"name169","name2":"name8539","long":"123456789012345678901234567890"}

// KEYS
using K_L1_name  = metastring("name");
using K_L1_name2 = metastring("name2");
using K_L1_long  = metastring("long");

// NOTE: dynamic string size are dynamic configurable, but not per field
template<class Key, int Size, class Options = boost::mp11::mp_list<>>
using JStringVariant = JStringStaticCopy<mp_int<Size>, Key, Options>;

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
#if 0
        mp_list<K_L1_name,  NullDefaultEmptyString<StringFun<K_L1_name, STATIC_STRING_SIZE>>>,
		mp_list<K_L1_name2, NullDefaultEmptyString<StringFun<K_L1_name2, STATIC_STRING_SIZE>>>,
		mp_list<K_L1_long,  NullDefaultEmptyString<StringFun<K_L1_long, STATIC_STRING_SIZE>>>
#else
#pragma message("Always using JStringStaticCopy for parsing strings")
		mp_list<K_L1_name,  NullDefaultEmptyString<JStringVariant<K_L1_name, STATIC_STRING_SIZE>>>,
		mp_list<K_L1_name2, NullDefaultEmptyString<JStringVariant<K_L1_name2, STATIC_STRING_SIZE>>>,
		mp_list<K_L1_long,  NullDefaultEmptyString<JStringVariant<K_L1_long, STATIC_STRING_SIZE>>>
#endif
>,
        DictOpts
> ;