#include <map>

// we only need public host functions and types for the CUDA runtime API;
// no built-in type definitions and device intrinsic functions needed
#include <cuda_runtime_api.h>

/**
 * Describe type of GPU or CPU memory
 *
 * Possible results:
 * - unregistered host memory
 * - registered host memory
 * - device memory
 * - managed memory
 *
 * Possible results in case of an error:
 * - No unified addressing
 * - Error retrieving attributes
 * - Unknown CUDA memory type
 *
 * @param ptr pointer to host or device memory
 * @return C string describing the type of memory, or cause of error (never NULL)
 */
const char* memory_desc(const void *ptr)
{
    const std::map<enum cudaMemoryType, const char*> memory_type_map
        {
         { cudaMemoryTypeUnregistered, "unregistered host memory" },
         { cudaMemoryTypeHost, "registered host memory" },
         { cudaMemoryTypeDevice, "device memory" },
         { cudaMemoryTypeManaged, "managed memory" }
        };

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        if (err == cudaErrorInvalidValue)
            return "No unified addressing";
        else
            return "Error retrieving attributes";
    }

    auto   it  = memory_type_map.find(attrs.type);
    return it == memory_type_map.end() ? "Unknown CUDA memory type" : it->second;
}
