#ifndef META_CUDF_PARSER_CUH
#define META_CUDF_PARSER_CUH
#include <cudf/io/types.hpp>

cudf::io::table_with_metadata generate_example_metadata(const char* filename, int count, cudaStream_t pStream);

#endif //META_CUDF_PARSER_CUH
