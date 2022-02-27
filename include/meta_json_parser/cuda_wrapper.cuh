#ifndef META_JSON_PARSER_CUDA_WRAPPER_CUH
#define META_JSON_PARSER_CUDA_WRAPPER_CUH

namespace template_cuda_math {
    template<class T>
    __device__ __forceinline__ T exp10(T value) { return "T only supported for float and double"; }

    template<>
    __device__ __forceinline__ float exp10<float>(float value) { return ::exp10f(value); }

    template<>
    __device__ __forceinline__ double exp10<double>(double value) { return ::exp10(value); }

    template<class T>
    __device__ __forceinline__ T floor(T value) { return "T only supported for float and double"; }

    template<>
    __device__ __forceinline__ float floor<float>(float value) { return ::floorf(value); }

    template<>
    __device__ __forceinline__ double floor<double>(double value) { return ::floor(value); }

    template<class T>
    __device__ __forceinline__ T nearbyint(T value) { return "T only supported for float and double"; }

    template<>
    __device__ __forceinline__ float nearbyint<float>(float value) { return ::nearbyintf(value); }

    template<>
    __device__ __forceinline__ double nearbyint<double>(double value) { return ::nearbyint(value); }

    template<class T>
    __device__ __forceinline__ T log10(T value) { return "T only supported for float and double"; }

    template<>
    __device__ __forceinline__ float log10<float>(float value) { return ::log10f(value); }

    template<>
    __device__ __forceinline__ double log10<double>(double value) { return ::log10(value); }

    template<class T>
    __device__ __forceinline__ T fabs(T value) { return "T only supported for float and double"; }

    template<>
    __device__ __forceinline__ float fabs<float>(float value) { return ::fabsf(value); }

    template<>
    __device__ __forceinline__ double fabs<double>(double value) { return ::fabs(value); }

    template<class T>
    __device__ __forceinline__ T frexp(T value, int* nptr) { return "T only supported for float and double"; }

    template<>
    __device__ __forceinline__ float frexp<float>(float value, int* nptr) { return ::frexpf(value, nptr); }

    template<>
    __device__ __forceinline__ double frexp<double>(double value, int* nptr) { return ::frexp(value, nptr); }
}

#endif //META_JSON_PARSER_CUDA_WRAPPER_CUH
