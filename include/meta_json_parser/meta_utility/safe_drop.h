#ifndef HASH_PY_SAFE_DROP_H
#define HASH_PY_SAFE_DROP_H
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/mp11/function.hpp>

template<class L, class N>
using safe_drop = boost::mp11::mp_eval_if_c<
    boost::mp11::mp_size<L>::value < N::value,
    boost::mp11::mp_list<>,
    boost::mp11::mp_drop,
    L,
    N
>;

template<class L, size_t N>
using safe_drop_c = safe_drop<L, boost::mp11::mp_int<N>>;

#endif //HASH_PY_SAFE_DROP_H
