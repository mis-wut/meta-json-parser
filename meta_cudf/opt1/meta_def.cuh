#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

// INCLUDES
#include <boost/mp11.hpp>

#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/meta_utility/metastring.h>

// TYPE-DEPENDENT INCLUDES
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jstring_custom.cuh>
#include <meta_json_parser/action/string_transform_functors/polynomial_rolling_hash.cuh>
//#include <meta_json_parser/action/jrealnumber.cuh>
//#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/action/jbool.cuh>

//#include <meta_json_parser/action/jstring_custom.cuh>
//#include <meta_json_parser/action/string_transform_functors/polynomial_rolling_hash_matcher.cuh>
#include <meta_json_parser/action/decorators/null_default_value.cuh>
#include <meta_json_parser/action/string_functors/letter_case.cuh>

using namespace boost::mp11;
using namespace std;

// SETTINGS
using WorkGroupSize = mp_int<32>;

// EXAMPLE:
/*
{
	"author": "Dethcola",                 // max length seen: 20
	"author_flair_css_class": "",         // max length seen: 61, min length seen: 0, nullable
	"author_flair_text": "Clairemont",    // max length seen: 55, min length seen: 0, nullable
	"body": "A quarry",                   // max length seen: 1775, min length seen: 2
	"can_gild": true,                     // bool, all true (?)
	"controversiality": 0,                // 0 or 1 in this sample, unsigned integer
	"created_utc": 1506816000,            // timestamp
	"distinguished": null,                // "moderator" or null
	"edited": false,                      // ??? timestamp or false
	"gilded": 0,                          // integer, all 0
	"id": "dnqik14",                      // 7 character identifier
	"is_submitter": false,                // bool
	"link_id": "t3_73ieyz",               // 9 character identifier, starting with "t3_" prefix
	"parent_id": "t3_73ieyz",             // 9 character identifier, starting with "t3_" prefix
	"permalink": "/r/sandiego/comments/73ieyz/best_place_for_granite_counter_tops/dnqik14/",  // max length: 95, min: 46
	"retrieved_on": 1509189606,           // timestamp
	"score": 3,                           // smallish integer, signed values, seen values range: -56..500
	"stickied": false,                    // bool, mostly false
	"subreddit": "sandiego",              // max length seen: 18, min length seen: 3
	"subreddit_id": "t5_2qq2q"            // 8 or 7 character identifier, starting with "t5_" prefix
}
*/

using Multiplier = std::integral_constant<uint64_t, 31>;
using Modulus = std::integral_constant<uint64_t, static_cast<uint64_t>(1e9 + 9)>;

template<class TagT>
using StringHash = JStringCustom<PolynomialRollingHashFunctor<Multiplier, Modulus, size_t, TagT>>;

// KEYS (Key, Level 1, ...)
using K_L1_author = metastring("author");
using K_L1_cakeday = metastring("author_cakeday");
using K_L1_flair_css = metastring("author_flair_css_class");
using K_L1_flair = metastring("author_flair_text");
using K_L1_body = metastring("body");
using K_L1_can_gild = metastring("can_gild");
using K_L1_controv = metastring("controversiality");
using K_L1_created_utc = metastring("created_utc");
using K_L1_distinguished = metastring("distinguished");
using K_L1_edited = metastring("edited");
using K_L1_gilded = metastring("gilded");
using K_L1_id = metastring("id");
using K_L1_is_submitter = metastring("is_submitter");
using K_L1_link_id = metastring("link_id");
using K_L1_parent_id = metastring("parent_id");
using K_L1_permalink = metastring("permalink");
using K_L1_retrieved_on = metastring("retrieved_on");
using K_L1_score = metastring("score");
using K_L1_stickied = metastring("stickied");
using K_L1_subreddit = metastring("subreddit");
using K_L1_subreddit_id = metastring("subreddit_id");

// DATETIME FORMATS
// datetimes are stored as timestamps (as Unix epoch)


#define USE_TRANSFORMATIONS 1
#ifdef USE_TRANSFORMATIONS
// CONFIGURE TRANSFORMATIONS
#define USE_STR_LOWER_TRANSFORMATION 1
#ifdef USE_STR_LOWER_TRANSFORMATION
#pragma message("Using df[<column>].str.lower transformation compile-time")
using JStringToLowerTransformConf = mp_list< // dict
	mp_list< // key: value
		JStringOptions::JStringCharTransformer,
		ToLowerStringTransformer
	>
>;
#endif
#endif

// CONFIGURE STRING PARSING
#pragma message("Always using JStringStaticCopy for parsing strings")
// NOTE: dynamic string size are dynamic configurable, but not per field
template<class Key, int Size, class Options = boost::mp11::mp_list<>>
using JStringVariant = JStringStaticCopy<mp_int<Size>, Key, Options>;

using SignedIntOpt = mp_list<
   mp_list<
       JNumberOptions::JNumberSign,
       JNumberOptions::JNumberSign::Signed
   >
>;



// DICT
#define STATIC_STRING_SIZE 32
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <

#ifdef USE_TRANSFORMATIONS
	mp_list<K_L1_author, StringHash<K_L1_author>>, // Hash
    	mp_list<K_L1_flair_css, NullDefaultInteger<StringHash<K_L1_flair_css>, mp_int<0>>>, // 0 if null, else hash
    	mp_list<K_L1_flair, NullDefaultInteger<StringHash<K_L1_flair>, mp_int<0>>>, // 0 if null, else hash
#else
	mp_list<K_L1_author, JStringVariant<K_L1_author, 64>>,
	mp_list<K_L1_flair_css, NullDefaultEmptyString<JStringVariant<K_L1_flair_css, 64>>>,
	mp_list<K_L1_flair, NullDefaultEmptyString<JStringVariant<K_L1_flair, 64>>>,
#endif
#ifdef USE_STR_LOWER_TRANSFORMATION
	mp_list<K_L1_body, JStringVariant<K_L1_body, 2048, JStringToLowerTransformConf>>,
#else
	mp_list<K_L1_body, JStringVariant<K_L1_body, 2048>>,
#endif
	mp_list<K_L1_can_gild, JBool<uint8_t, K_L1_can_gild>>, // NOTE: must be uint8_t
	mp_list<K_L1_controv, JNumber<uint32_t, K_L1_controv>>, // NOTE: uint16_t would be enough
	mp_list<K_L1_created_utc, JNumber<int64_t, K_L1_created_utc>>, // NOTE: timestamp, use int64_t for easy conversion
	mp_list<K_L1_distinguished, NullDefaultInteger<JNumber<int64_t, K_L1_distinguished>, mp_int<0>>>,
	//mp_list<K_L1_edited, JBool<uint8_t, K_L1_edited>>, // NOTE: must be uint8_t; NOTE: data needs fixing !!!
	mp_list<K_L1_edited, JNumber<uint32_t, K_L1_edited>>, // NOTE: must be uint8_t; NOTE: data needs fixing !!!
	mp_list<K_L1_gilded, JNumber<uint32_t, K_L1_gilded>>, // NOTE: uint16_t would be enough
	mp_list<K_L1_id, JStringVariant<K_L1_id, 32>>,
	mp_list<K_L1_is_submitter, JBool<uint8_t, K_L1_is_submitter>>, // NOTE: must be uint8_t
	mp_list<K_L1_link_id, JStringVariant<K_L1_link_id, 32>>,
	mp_list<K_L1_parent_id, JStringVariant<K_L1_parent_id, 32>>,
	mp_list<K_L1_permalink, JStringVariant<K_L1_permalink, 128>>,
	mp_list<K_L1_score, JNumber<int32_t, K_L1_score, SignedIntOpt>>, // NOTE: signed, int16_t could be enough
	mp_list<K_L1_stickied, JBool<uint8_t, K_L1_stickied>>, // NOTE: must be uint8_t
#ifdef USE_TRANSFORMATIONS
	mp_list<K_L1_subreddit, StringHash<K_L1_subreddit>>,
#else
	mp_list<K_L1_subreddit, JStringVariant<K_L1_subreddit, 32>>,
#endif
	mp_list<K_L1_subreddit_id, JStringVariant<K_L1_subreddit_id, 32>>,
	mp_list<K_L1_retrieved_on, JNumber<int64_t, K_L1_retrieved_on>> // NOTE: timestamp, use int64_t for easy conversion
>,
    DictOpts
> ;

// NOTE: Neither PARSER OPTIONS nor PARSER are needed for 'data_def.cuh'
// that is for inclusion in the 'benchmark/main.cu'
#ifndef BENCHMARK_MAIN_CU

// PARSER OPTIONS
template<class Key, int Size>
using StaticCopyFun = JStringStaticCopy<mp_int<Size>, Key>;

// PARSER
using BaseAction = DictCreator<StaticCopyFun, mp_list<>>;
#endif /* !defined(BENCHMARK_MAIN_CU) */

#endif /* !defined(META_CUDF_META_DEF_CUH) */
