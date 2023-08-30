#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>

#warning "Using methods and classes in cudf::test:: namespace"
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include "generate_dataframe.h"

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using str_col_wrapper = cudf::test::strings_column_wrapper;

cudf::io::table_with_metadata generate_example_dataframe() {
  // 1. create individual columns, each with the same number of rows
  std::cout << "creating 6 individual columns, 4 rows each...\n";
  std::cout << "creating 6 individual columns, 4 rows each...\n";
  column_wrapper<int32_t> col1{{4, 5, 6, 7}};
  column_wrapper<float> col2{
      {-0.0f, 1e-5f, 1.2345f, std::numeric_limits<float>::infinity()}};
  column_wrapper<bool> col3{{true, false, true, false}};
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s,
                                         cudf::timestamp_s::rep>
      col4{{0, 798163200, 1622060798, 2764800}};
  str_col_wrapper col5{
      {"1.,;	|?  ", "abc def ghi", "\"jkl mno pqr\"", "stu \"wv\" xyz"}};
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns, int64_t> col6{
      {-86400L, -1L, 0L, 286134307L}};

  // 2. create a vector of all columns of the table
  std::cout << "creating vector of columns...\n";
  std::vector<std::unique_ptr<cudf::column>> columns;

  // 3. add columns to vector of columns
  std::cout << "emplacing 6 columns...\n";
  columns.emplace_back(col1.release());
  columns.emplace_back(col2.release());
  columns.emplace_back(col3.release());
  columns.emplace_back(col4.release());
  columns.emplace_back(col5.release());
  columns.emplace_back(col6.release());

  // 4. create a table (which will be turned into DataFrame equivalent)
  std::cout << "creating table...\n";
  cudf::table table{std::move(columns)}; // std::move or std::forward
  std::cout << "...with " << table.num_columns() << " columns and "
            << table.num_rows() << " rows\n";

  // 5. create a wrapper with names of columns
  std::cout << "creating vector of column names...\n";
  std::vector<std::string> column_names{
      "int32", "float", "bool", "timestamp[s]", "string", "duration[ns]"};
  std::cout << "...with " << column_names.size() << " elements\n";
  std::cout << "...{";
  std::copy(column_names.cbegin(), column_names.cend(),
            std::ostream_iterator<std::string>(std::cout, ","));
  std::cout << "}\n";
  std::cout << "creating metadata for table (table_metadata)...\n";
  cudf::io::table_metadata metadata{column_names};

  // 6. create table_with_metadata to return
  std::cout << "creating table with metadata...\n";
  cudf::io::table_with_metadata result{std::make_unique<cudf::table>(table),
                                       metadata};
  std::cout << "(done)\n";

  return result;
}
