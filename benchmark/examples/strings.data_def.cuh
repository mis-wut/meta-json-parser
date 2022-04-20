// INCLUDES
#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"name":"name169","name2":"name8539","long":"123456789012345678901234567890"}

// KEYS
using K_L1_name  = metastring("name");
using K_L1_name2 = metastring("name2");
using K_L1_long  = metastring("long");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_name,  StringFun<K_L1_name, STATIC_STRING_SIZE>>,
		mp_list<K_L1_name2, StringFun<K_L1_name2, STATIC_STRING_SIZE>>,
		mp_list<K_L1_long,  StringFun<K_L1_long, STATIC_STRING_SIZE>>
>,
        DictOpts
> ;