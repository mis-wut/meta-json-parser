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
