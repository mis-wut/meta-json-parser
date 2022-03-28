#ifndef MEDA_CUDF_META_DEF_CUH
#define MEDA_CUDF_META_DEF_CUH
#include <boost/mp11.hpp>
#include <meta_json_parser/action/jnumber.cuh>

using WorkGroupSize = boost::mp11::mp_int<32>;

using BaseAction = JNumber<int, void>;

#endif //META_CUDF_META_DEF_CUH
