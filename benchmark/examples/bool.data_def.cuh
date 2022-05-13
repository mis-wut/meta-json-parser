// INCLUDES
#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <meta_json_parser/action/jbool.cuh>
#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"is_checked":false,"1_is_checked":true,"2_is_checked":true}

// KEYS
using K_L1_is_checked   = metastring("is_checked");
using K_L1_1_is_checked = metastring("1_is_checked");
using K_L1_2_is_checked = metastring("2_is_checked");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_is_checked,   JBool<uint8_t, K_L1_is_checked>>,
		mp_list<K_L1_1_is_checked, JBool<uint8_t, K_L1_1_is_checked>>,
		mp_list<K_L1_2_is_checked, JBool<uint8_t, K_L1_2_is_checked>>, 
>,
        DictOpts
> ;

#endif /* !defined(META_CUDF_META_DEF_CUH) */