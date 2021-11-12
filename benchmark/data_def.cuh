
// KEYS
using K_L1_date = mp_string<'d', 'a', 't', 'e'>;
using K_L1_lat = mp_string<'l', 'a', 't'>;
using K_L1_lon = mp_string<'l', 'o', 'n'>;
using K_L1_is_checked = mp_string<'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_name = mp_string<'n', 'a', 'm', 'e'>;
using K_L1_1_date = mp_string<'1', '_', 'd', 'a', 't', 'e'>;
using K_L1_1_lat = mp_string<'1', '_', 'l', 'a', 't'>;
using K_L1_1_lon = mp_string<'1', '_', 'l', 'o', 'n'>;
using K_L1_1_is_checked = mp_string<'1', '_', 'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_1_name = mp_string<'1', '_', 'n', 'a', 'm', 'e'>;
using K_L1_2_date = mp_string<'2', '_', 'd', 'a', 't', 'e'>;
using K_L1_2_lat = mp_string<'2', '_', 'l', 'a', 't'>;
using K_L1_2_lon = mp_string<'2', '_', 'l', 'o', 'n'>;
using K_L1_2_is_checked = mp_string<'2', '_', 'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_2_name = mp_string<'2', '_', 'n', 'a', 'm', 'e'>;
using K_L1_3_date = mp_string<'3', '_', 'd', 'a', 't', 'e'>;
using K_L1_3_lat = mp_string<'3', '_', 'l', 'a', 't'>;
using K_L1_3_lon = mp_string<'3', '_', 'l', 'o', 'n'>;
using K_L1_3_is_checked = mp_string<'3', '_', 'i', 's', '_', 'c', 'h', 'e', 'c', 'k', 'e', 'd'>;
using K_L1_3_name = mp_string<'3', '_', 'n', 'a', 'm', 'e'>;

// DICT
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_date, StringFun<K_L1_date, 32>>,
        mp_list<K_L1_lat, JNumber<uint32_t, K_L1_lat>>,
        mp_list<K_L1_lon, JNumber<uint32_t, K_L1_lon>>,
        mp_list<K_L1_is_checked, JBool<uint8_t, K_L1_is_checked>>,
        mp_list<K_L1_name, StringFun<K_L1_name, 32>>,
        mp_list<K_L1_1_date, StringFun<K_L1_1_date, 32>>,
        mp_list<K_L1_1_lat, JNumber<uint32_t, K_L1_1_lat>>,
        mp_list<K_L1_1_lon, JNumber<uint32_t, K_L1_1_lon>>,
        mp_list<K_L1_1_is_checked, JBool<uint8_t, K_L1_1_is_checked>>,
        mp_list<K_L1_1_name, StringFun<K_L1_1_name, 32>>,
        mp_list<K_L1_2_date, StringFun<K_L1_2_date, 32>>,
        mp_list<K_L1_2_lat, JNumber<uint32_t, K_L1_2_lat>>,
        mp_list<K_L1_2_lon, JNumber<uint32_t, K_L1_2_lon>>,
        mp_list<K_L1_2_is_checked, JBool<uint8_t, K_L1_2_is_checked>>,
        mp_list<K_L1_2_name, StringFun<K_L1_2_name, 32>>,
        mp_list<K_L1_3_date, StringFun<K_L1_3_date, 32>>,
        mp_list<K_L1_3_lat, JNumber<uint32_t, K_L1_3_lat>>,
        mp_list<K_L1_3_lon, JNumber<uint32_t, K_L1_3_lon>>,
        mp_list<K_L1_3_is_checked, JBool<uint8_t, K_L1_3_is_checked>>,
        mp_list<K_L1_3_name, StringFun<K_L1_3_name, 32>>
>,
        DictOpts
> ;
