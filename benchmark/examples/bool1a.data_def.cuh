// INCLUDES
#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <meta_json_parser/action/jbool.cuh>
#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"is_checked":false}

// KEYS
using K_L1_is_checked   = metastring("a");

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_is_checked,   JBool<uint8_t, K_L1_is_checked>> 
>,
        DictOpts
> ;

#ifdef HAVE_LIBCUDF
#define HAVE_DTYPES
std::map< std::string, cudf::data_type > dtypes{
    { "a",   cudf::data_type{cudf::type_id::BOOL8} },
};
#endif

#endif /* !defined(META_CUDF_META_DEF_CUH) */