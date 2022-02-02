cmake_minimum_required(VERSION 3.18)

project(meta-json-parser VERSION 0.1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 10.0 REQUIRED)

IF(NOT LOCAL_LIB)
find_package(Boost 1.73)  # version 1.73 is required to use boost::mp11::mp_flatten
ENDIF()

IF(NOT Boost_FOUND)
    add_subdirectory(third_parties/mp11)
    add_library(Boost::boost ALIAS boost_mp11)
ENDIF()

IF(NOT LOCAL_LIB)
find_package(GTest REQUIRED CONFIG)
ENDIF()

IF(NOT GTest_FOUND)
    add_subdirectory(third_parties/googletest)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gmock ALIAS gmock)
ENDIF()

include(GoogleTest)

add_library(lib-meta-json-parser INTERFACE)

target_compile_features(lib-meta-json-parser INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(lib-meta-json-parser INTERFACE "${PROJECT_SOURCE_DIR}/include")

target_sources(lib-meta-json-parser INTERFACE
    include/meta_json_parser/action/jarray.cuh
    include/meta_json_parser/action/jbool.cuh
    include/meta_json_parser/action/jdict.cuh
    include/meta_json_parser/action/jnumber.cuh
    include/meta_json_parser/action/jstring.cuh
    include/meta_json_parser/action/jstring_custom.cuh
    include/meta_json_parser/action/string_functors/letter_case_functors.cuh
    include/meta_json_parser/action/void_action.cuh
    include/meta_json_parser/action_iterator.h
    include/meta_json_parser/byte_algorithms.h
    include/meta_json_parser/config.h
    include/meta_json_parser/cub_wrapper.cuh
    include/meta_json_parser/intelisense_silencer.h
    include/meta_json_parser/json_parse.cuh
    include/meta_json_parser/kernel_context.cuh
    include/meta_json_parser/kernel_launch_configuration.cuh
    include/meta_json_parser/kernel_launcher.cuh
    include/meta_json_parser/memory/allocator.h
    include/meta_json_parser/memory/config.h
    include/meta_json_parser/memory/config_transform.h
    include/meta_json_parser/memory/pool.h
    include/meta_json_parser/memory_configuration.h
    include/meta_json_parser/memory_request.h
    include/meta_json_parser/meta_math.h
    include/meta_json_parser/meta_memory_manager.cuh
    include/meta_json_parser/meta_utils.h
    include/meta_json_parser/mp_string.h
    include/meta_json_parser/output_manager.cuh
    include/meta_json_parser/output_printer.cuh
    include/meta_json_parser/parse.cuh
    include/meta_json_parser/parser_configuration.h
    include/meta_json_parser/parser_input.h
    include/meta_json_parser/parser_kernel.cuh
    include/meta_json_parser/parser_output_device.cuh
    include/meta_json_parser/parser_output_host.cuh
    include/meta_json_parser/parser_requirements.cuh
    include/meta_json_parser/parsing_error.h
    include/meta_json_parser/runtime_configuration.cuh
    include/meta_json_parser/static_buffer.h
    include/meta_json_parser/work_group_reader.cuh
)

target_link_libraries(lib-meta-json-parser INTERFACE Boost::boost CUDA::cudart nvToolsExt)

add_executable(meta-json-parser-test)

target_link_libraries(meta-json-parser-test PRIVATE lib-meta-json-parser GTest::gtest GTest::gmock)

target_sources(meta-json-parser-test PRIVATE
    include/meta_json_parser/checkpoint_results.cpp
    include/meta_json_parser/checkpoint_results.h
    include/meta_json_parser/debug_helpers.cpp
    test/book_json.cu
    test/keywriter_tests.cu
    test/main.cu
    test/parse_dynamic_string.cu
    test/parse_jarray.cu
    test/parse_jbool.cu
    test/parse_jdict.cu
    test/parse_jnumber.cu
    test/parse_jstring.cu
    test/parse_jstring_custom.cu
    test/parse_jstring_transform.cu
    test/parse_uint_test.cu
    test/read_only_buffer_test.cu
    test/test_configuration.h
    test/test_helper.h
    test/test_memory_config.cu
    test/test_utility/contexts/static_string_test_context.cu
    test/test_utility/contexts/static_string_test_context.cuh
    test/test_utility/contexts/string_test_context.cu
    test/test_utility/contexts/string_test_context.cuh
    test/test_utility/contexts/test_context.cu
    test/test_utility/contexts/test_context.cuh
    test/test_utility/operators.cuh
    test/test_utility/test_launcher.cu
    test/test_utility/test_launcher.cuh
    test/uint_test_context.cuh
    test/work_group_reader_test.cu
)

#set_target_properties(lib-meta-json-parser meta-json-parser-test PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 52)

target_compile_features(meta-json-parser-test INTERFACE cxx_std_17 cuda_std_17)

target_compile_options(meta-json-parser-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

enable_testing()

gtest_discover_tests(meta-json-parser-test)

# benchmarks
add_subdirectory(third_parties/CLI11)

#add_executable(meta-json-parser-benchmark benchmark/checkpoint_results.cpp benchmark/checkpoint_results.h)
add_executable(meta-json-parser-benchmark)

target_link_libraries(meta-json-parser-benchmark PRIVATE lib-meta-json-parser CLI11)

# libcudf is an optional dependency for meta-json-parser-benchmark
# run cmake with -DUSE_LIBCUDF=1 to enable it
if (USE_LIBCUDF)
    message("USE_LIBCUDF is true, compiling with libcudf support")
    # TODO: make location of the cudf library and its prerequisties configurable
    #       instead of hardcoding paths to includes and to libraries, like below
    target_include_directories(meta-json-parser-benchmark PUBLIC
                               /usr/local/include
                               /usr/local/include/libcudf/libcudacxx
                               /opt/conda/envs/rapids/include
    )
    target_link_libraries(meta-json-parser-benchmark PRIVATE
                          cudf
                          -L/usr/local/cuda/lib64 -L/usr/local/lib
                          -L/opt/conda/envs/rapids/lib
    )
    target_compile_definitions(meta-json-parser-benchmark PRIVATE
                               HAVE_LIBCUDF=1
    )
endif()

target_sources(meta-json-parser-benchmark PRIVATE
	benchmark/main.cu
	include/meta_json_parser/debug_helpers.cpp
	include/meta_json_parser/checkpoint_results.cpp
	include/meta_json_parser/checkpoint_results.h
)

target_compile_features(meta-json-parser-benchmark INTERFACE cxx_std_17 cuda_std_17)

target_compile_options(meta-json-parser-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(meta-json-parser-benchmark PRIVATE
                      nvToolsExt  # for easier profiling of meta-json-parser-benchmark executable
)
