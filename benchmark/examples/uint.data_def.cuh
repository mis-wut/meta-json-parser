// corresponding JSON file: uint.json
// { "uint": 16 }

// KEYS
using K_L1_uint = mp_string<'u', 'i', 'n', 't'>;

// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
    mp_list<K_L1_uint, JNumber<uint32_t, K_L1_uint>>
>,
    DictOpts
> ;
