cimport cudf._lib.utils
cimport cudf._lib.cpp.io.types as cudf_io_types

from cudf._lib.utils cimport data_from_unique_ptr
from libcpp.utility cimport move

cdef extern from "parser.cuh":
    cudf_io_types.table_with_metadata generate_example_metadata(const char* filename, int count);

def wrapped_test(fname, count, pStream):

    cdef cudf_io_types.table_with_metadata c_out_table
    #c_out_table =  generate_example_metadata()

    column_names = [x.decode() for x in c_out_table.metadata.column_names]
    return data_from_unique_ptr(move(c_out_table.tbl),
                               column_names=column_names)
