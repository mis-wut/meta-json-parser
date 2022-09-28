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

using Multiplier = std::integral_constant<uint64_t, 31>;
using Modulus = std::integral_constant<uint64_t, static_cast<uint64_t>(1e9 + 9)>;

template<class TagT>
using StringHash = JStringCustom<PolynomialRollingHashFunctor<Multiplier, Modulus, size_t, TagT>>;

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

// Schema definition
#include "meta_def_schema.cuh"


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
