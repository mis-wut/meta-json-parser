#pragma once

#if defined(HAVE_LIBCUDF)
#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>
#endif /* HAVE_LIBCUDF */

const char* memory_desc(const void *ptr);

#if defined(HAVE_LIBCUDF)
const char* type_id_to_name(cudf::type_id id);
void column_data_dumper(const void *data, size_t data_size,
                        cudf::type_id type_id, int n_elems,
                        const char *indent = "");
void describe_column(cudf::column_view col, bool dump_data = false, const char *indent = "");
#endif /* HAVE_LIBCUDF */