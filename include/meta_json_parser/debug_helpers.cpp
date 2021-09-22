#include <map>
#include <cstdio>
#include <iostream>

// should it be cuda_runtime_api.h or cuda_runtime.h?
#include <cuda_runtime_api.h>

// TODO: make configurable with CMake
#define HAVE_LIBCUDF
#if defined(HAVE_LIBCUDF)
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/traits.hpp>
#endif /* HAVE_LIBCUDF */

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


#if defined(HAVE_LIBCUDF)
const char* type_id_to_name(cudf::type_id id)
{
    const std::map<cudf::type_id, const char*> type_name_map
        {
         { cudf::type_id::EMPTY, "cudf::type_id::EMPTY" },
         { cudf::type_id::INT8, "cudf::type_id::INT8" },
         { cudf::type_id::INT16, "cudf::type_id::INT16" },
         { cudf::type_id::INT32, "cudf::type_id::INT32" },
         { cudf::type_id::INT64, "cudf::type_id::INT64" },
         { cudf::type_id::UINT8, "cudf::type_id::UINT8" },
         { cudf::type_id::UINT16, "cudf::type_id::UINT16" },
         { cudf::type_id::UINT32, "cudf::type_id::UINT32" },
         { cudf::type_id::UINT64, "cudf::type_id::UINT64" },
         { cudf::type_id::FLOAT32, "cudf::type_id::FLOAT32" },
         { cudf::type_id::FLOAT64, "cudf::type_id::FLOAT64" },
         { cudf::type_id::BOOL8, "cudf::type_id::BOOL8" },
         { cudf::type_id::TIMESTAMP_DAYS, "cudf::type_id::TIMESTAMP_DAYS" },
         { cudf::type_id::TIMESTAMP_SECONDS, "cudf::type_id::TIMESTAMP_SECONDS" },
         { cudf::type_id::TIMESTAMP_MILLISECONDS, "cudf::type_id::TIMESTAMP_MILLISECONDS" },
         { cudf::type_id::TIMESTAMP_MICROSECONDS, "cudf::type_id::TIMESTAMP_MICROSECONDS" },
         { cudf::type_id::TIMESTAMP_NANOSECONDS, "cudf::type_id::TIMESTAMP_NANOSECONDS" },
         { cudf::type_id::DURATION_DAYS, "cudf::type_id::DURATION_DAYS" },
         { cudf::type_id::DURATION_SECONDS, "cudf::type_id::DURATION_SECONDS" },
         { cudf::type_id::DURATION_MILLISECONDS, "cudf::type_id::DURATION_MILLISECONDS" },
         { cudf::type_id::DURATION_MICROSECONDS, "cudf::type_id::DURATION_MICROSECONDS" },
         { cudf::type_id::DURATION_NANOSECONDS, "cudf::type_id::DURATION_NANOSECONDS" },
         { cudf::type_id::DICTIONARY32, "cudf::type_id::DICTIONARY32" },
         { cudf::type_id::STRING, "cudf::type_id::STRING" },
         { cudf::type_id::LIST, "cudf::type_id::LIST" },
         { cudf::type_id::DECIMAL32, "cudf::type_id::DECIMAL32" },
         { cudf::type_id::DECIMAL64, "cudf::type_id::DECIMAL64" },
         { cudf::type_id::STRUCT, "cudf::type_id::STRUCT" }
        };
    auto   it  = type_name_map.find(id);
    return it == type_name_map.end() ? "Unknown cudf::type_id" : it->second;
}
#endif /* HAVE_LIBCUDF */


// TODO: perhaps move this part to a separate module / separate file
#if defined(HAVE_LIBCUDF)
void column_data_dumper(const void *data, size_t data_size,
                        cudf::type_id type_id, int n_elems,
                        const char *indent = "")
{
    char *h_data = (char *)malloc(data_size);
    cudaMemcpy(h_data, data, data_size, cudaMemcpyDeviceToHost);

    printf("%s- host data pointer:         %p %s (size = %zd bytes)\n", indent,
           h_data, h_data ? memory_desc(h_data) : "", data_size);
    
    int32_t *h_data_int32 = (int32_t *)h_data;
    int64_t *h_data_int64 = (int64_t *)h_data;
    switch (type_id) {
    case cudf::type_id::INT32:
        std::cout << "::VALUES:: = [";
        std::copy(h_data_int32, h_data_int32 + n_elems,
                  std::ostream_iterator<int32_t>(std::cout, ", "));
        std::cout << "]\n";
        break;
        
    case cudf::type_id::INT64:
        std::cout << "::VALUES:: = [";
        std::copy(h_data_int64, h_data_int64 + n_elems,
                  std::ostream_iterator<int64_t>(std::cout, ", "));
        std::cout << "]\n";
        break;
        
    default:
        break;
    }
    //std::cout << "::DUMP:: " << data_size << "\n"
    //          << hexDump(h_data, data_size) << "\n";
    free(h_data);
}

void describe_column(cudf::column_view col, bool dump_data = false, const char *indent = "")
{
    const void *data = col.data<uint8_t>();
    bool nullable = col.nullable();
    int num_children = col.num_children();

    //cudf::id_to_type_impl<col.type().id()>::type var; // how to ue that ???
    
    printf("%s- number of elements:        %d\n", indent, col.size());
    printf("%s- data pointer:              %p %s\n", indent,
           data, data ? memory_desc(data) : "");
    printf("%s- can contain null elements: %s\n", indent, nullable ? "true" : "false");
    if (nullable) {
        printf("%s- count of null elements:    %d\n", indent, col.null_count());
    }

    printf("%s- data type of elements:     %s\n", indent, type_id_to_name(col.type().id()));
    if (cudf::is_fixed_width(col.type())) {
        printf("%s- size of elements:          %zd\n",indent, cudf::size_of(col.type()));
    } else {
        printf("%s- elements are variable width\n", indent);
    }
    if (data) {
        size_t data_size = col.size()*cudf::size_of(col.type());

        if (dump_data)
            column_data_dumper(data, data_size, col.type().id(), col.size(), indent);
    }
    printf("%s- number of child columns:   %d\n", indent, num_children);

    // TODO: very inefficient
    std::string child_indent = std::string(indent) + "    ";
    for (int child_idx = 0; child_idx < num_children; child_idx++) {
        auto child = col.child(child_idx);

        printf("  - child(%d):\n", child_idx);
        describe_column(child, dump_data, child_indent.c_str());
    }
}
#endif /* HAVE_LIBCUDF */