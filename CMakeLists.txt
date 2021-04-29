cmake_minimum_required(VERSION 3.18)

project(meta-json-parser VERSION 0.1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 10.0 REQUIRED)

find_package(Boost 1.66 REQUIRED)

find_package(GTest REQUIRED)

add_library(lib-meta-json-parser INTERFACE)

target_compile_features(lib-meta-json-parser INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(lib-meta-json-parser INTERFACE "${PROJECT_SOURCE_DIR}/include")

target_sources(lib-meta-json-parser INTERFACE
	include/meta_json_parser/byte_algorithms.h
	include/meta_json_parser/config.h
	include/meta_json_parser/cub_wrapper.cuh
	include/meta_json_parser/intelisense_silencer.h
	include/meta_json_parser/json_parse.cuh
	include/meta_json_parser/kernel_context.cuh
	include/meta_json_parser/kernel_launcher.cuh
	include/meta_json_parser/memory_configuration.h
	include/meta_json_parser/memory_request.h
	include/meta_json_parser/meta_math.h
	include/meta_json_parser/meta_memory_manager.cuh
	include/meta_json_parser/output_manager.cuh
	include/meta_json_parser/parse.cuh
	include/meta_json_parser/parser_configuration.h
	include/meta_json_parser/parser_kernel.cuh
	include/meta_json_parser/parsing_error.h
	include/meta_json_parser/runtime_configuration.cuh
	include/meta_json_parser/static_buffer.h
	include/meta_json_parser/work_group_reader.cuh
	include/meta_json_parser/action/jarray.cuh
	include/meta_json_parser/action/jnumber.cuh
	include/meta_json_parser/action/jstring.cuh
)

target_link_libraries(lib-meta-json-parser INTERFACE Boost::boost CUDA::cudart)

add_executable(meta-json-parser-test)

target_link_libraries(meta-json-parser-test PRIVATE lib-meta-json-parser GTest::gtest)

target_sources(meta-json-parser-test PRIVATE
	test/main.cu
	test/parse_jarray.cu
	test/parse_jnumber.cu
	test/parse_jstring.cu
	test/parse_uint_test.cu
	test/uint_test_context.cuh
	test/work_group_reader_test.cu
)

#set_target_properties(lib-meta-json-parser meta-json-parser-test PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 52)

target_compile_features(meta-json-parser-test INTERFACE cxx_std_17 cuda_std_17)

target_compile_options(meta-json-parser-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

#enable_testing()

gtest_discover_tests(meta-json-parser-test)

# benchmarks
add_executable(meta-json-parser-benchmark)

target_link_libraries(meta-json-parser-benchmark PRIVATE lib-meta-json-parser GTest::gtest)

target_sources(meta-json-parser-benchmark PRIVATE
	benchmark/main.cu
)

target_compile_features(meta-json-parser-benchmark INTERFACE cxx_std_17 cuda_std_17)

target_compile_options(meta-json-parser-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
