// INCLUDES
#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"l1":{"l2":{"l3":800}}}

// generated using poc/generate_nested_pos.py from json2meta

// KEYS
using K_L1  = metastring("l1");
using K_L2  = metastring("l2");
using K_L3  = metastring("l3");
//using K_L4  = metastring("l4");
//using K_L5  = metastring("l5");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1,
            JDict < mp_list <
                mp_list<K_L2, 
                    JDict < mp_list <
                        mp_list<K_L3, JNumber<uint32_t, K_L3>>
                    >, DictOpts >
                >
            >, DictOpts >
        >
>,
        DictOpts
> ;


#endif /* !defined(META_CUDF_META_DEF_CUH) */