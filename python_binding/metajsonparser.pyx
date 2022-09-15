cimport cudf._lib.utils
cimport cudf._lib.cpp.io.types as cudf_io_types

import cudf
from cudf._lib.utils cimport data_from_unique_ptr
from libcpp.utility cimport move

cdef extern from "parser.cuh":
    cudf_io_types.table_with_metadata generate_example_metadata(const char* filename, int count);

def read_json(fname: str, count: int):

    cdef cudf_io_types.table_with_metadata c_out_table
    py_byte_string = fname.encode('ASCII')
    cdef const char* c_string = py_byte_string
    print(fname, count)
    c_out_table =  generate_example_metadata(c_string, count)

    column_names = [x.decode() for x in c_out_table.metadata.column_names]
    df = data_from_unique_ptr(move(c_out_table.tbl), column_names=column_names)
    return cudf.DataFrame._from_data(*df)
