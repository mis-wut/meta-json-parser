#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/integral.hpp>

using AllWorkGroups = boost::mp11::mp_list<
    boost::mp11::mp_int<32>,
    boost::mp11::mp_int<16>,
    boost::mp11::mp_int<8>,
    boost::mp11::mp_int<4>
>;

struct WorkGroupNameGenerator {
    template <typename T>
    static std::string GetName(int i) {
        std::stringstream stream;
        stream << "WS_" << T::value;
        return stream.str();
    }
};

#define META_WS_32(mfun, ...)\
mfun(32 __VA_OPT__(,) __VA_ARGS__)

#define META_WS_16(mfun, ...)\
META_WS_32(mfun __VA_OPT__(,) __VA_ARGS__)\
mfun(16 __VA_OPT__(,) __VA_ARGS__)

#define META_WS_8(mfun, ...)\
META_WS_16(mfun __VA_OPT__(,) __VA_ARGS__)\
mfun(8 __VA_OPT__(,) __VA_ARGS__)

#define META_WS_4(mfun, ...)\
META_WS_8(mfun __VA_OPT__(,) __VA_ARGS__)\
mfun(4 __VA_OPT__(,) __VA_ARGS__)

#define META_WS_2(mfun, ...)\
META_WS_4(mfun __VA_OPT__(,) __VA_ARGS__)\
mfun(2 __VA_OPT__(,) __VA_ARGS__)

#define META_WS_1(mfun)\
META_WS_2(mfun)\
mfun(1)
