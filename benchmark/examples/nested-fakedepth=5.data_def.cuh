// INCLUDES
#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"l1_____l2_____l3_____l4_____l5":800}
//
// NOTE: the same length as
// {"l1":{"l2":{"l3":{"l4":{"l5":800}}}}}

// generated using the following command:
// $ cat nested-depth\=5_1000000.jsonl |\ 
//   jq -c '{"l1_____l2_____l3_____l4_____l5":.l1.l2.l3.l4.l5}' \ 
//   >nested-fakedepth\=5_1000000.jsonl
// where nested-depth\=5_1000000.jsonl was generated using
// poc/generate_nested_pos.py script from json2meta repository

// KEYS
using K_L1  = metastring("l1_____l2_____l3_____l4_____l5");
//using K_L2  = metastring("l2");
//using K_L3  = metastring("l3");
//using K_L4  = metastring("l4");
//using K_L5  = metastring("l5");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1, JNumber<uint32_t, K_L1>>
>,
        DictOpts
> ;


#endif /* !defined(META_CUDF_META_DEF_CUH) */