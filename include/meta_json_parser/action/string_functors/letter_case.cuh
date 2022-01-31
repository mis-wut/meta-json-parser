#ifndef META_JSON_PARSER_TO_UPPER_CUH
#define META_JSON_PARSER_TO_UPPER_CUH
#include <cuda_runtime_api.h>
struct ToUpperStringTransformer {
    inline __device__ char operator()(char c) const {
        if (c >= 'a' && c <= 'z')
            c += 'A' - 'a';
        return c;
    }
};

struct ToLowerStringTransformer {
    inline __device__ char operator()(char c) const {
        if (c >= 'A' && c <= 'Z')
            c -= 'A' - 'a';
        return c;
    }
};
#endif //META_JSON_PARSER_TO_UPPER_CUH
