#pragma once
#ifndef GENERATE_DATAFRAME_H
#define GENERATE_DATAFRAME_H

#include <cudf/io/types.hpp>

// TODO: pass number of rows as parameter
cudf::io::table_with_metadata generate_example_dataframe();

#endif /* !defined(GENERATE_DATAFRAME_H) */
