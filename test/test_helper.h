#define META_WS_32(mfun)\
mfun(32)

#define META_WS_16(mfun)\
META_WS_32(mfun)\
mfun(16)

#define META_WS_8(mfun)\
META_WS_16(mfun)\
mfun(8)

#define META_WS_4(mfun)\
META_WS_8(mfun)\
mfun(4)

#define META_WS_2(mfun)\
META_WS_4(mfun)\
mfun(2)

#define META_WS_1(mfun)\
META_WS_2(mfun)\
mfun(1)
