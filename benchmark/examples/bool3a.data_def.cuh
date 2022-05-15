// INCLUDES
#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <meta_json_parser/action/jbool.cuh>
#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"a":false,"b":true,"c":false}

// KEYS
using K_L1_a = metastring("a");
using K_L1_b = metastring("b");
using K_L1_c = metastring("c");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_a, JBool<uint8_t, K_L1_a>>,
        mp_list<K_L1_b, JBool<uint8_t, K_L1_b>>,
		mp_list<K_L1_c, JBool<uint8_t, K_L1_c>> 
>,
        DictOpts
> ;

#ifdef HAVE_LIBCUDF
#define HAVE_DTYPES
std::map< std::string, cudf::data_type > dtypes{
    { "a", cudf::data_type{cudf::type_id::BOOL8} },
    { "b", cudf::data_type{cudf::type_id::BOOL8} },
    { "c", cudf::data_type{cudf::type_id::BOOL8} },
};
#endif /* defined(HAVE_LIBCUDF) */

#endif /* !defined(META_CUDF_META_DEF_CUH) */