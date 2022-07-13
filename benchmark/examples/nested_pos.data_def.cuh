// INCLUDES
#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

#include <meta_json_parser/action/jbool.cuh>
#include <meta_json_parser/meta_utility/metastring.h>

// EXAMPLE:
// {"pos":{"lat":-8.667,"lon":-174.078}}

// generated using poc/generate_nested_pos.py from json2meta

// KEYS
using K_L1_pos  = metastring("pos");
using K_L2_lat  = metastring("lat");
using K_L2_lon  = metastring("lon"); 

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_pos,
            JDict < mp_list <
                mp_list<K_L2_lat, JRealNumber<float, K_L2_lat>>,
                mp_list<K_L2_lon, JRealNumber<float, K_L2_lon>>
            >, DictOpts >
        >
>,
        DictOpts
> ;


#endif /* !defined(META_CUDF_META_DEF_CUH) */