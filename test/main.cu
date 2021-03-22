#include <gtest/gtest.h>

#include <boost/mp11/algorithm.hpp>
#include <meta_json_parser/action/jarray.cuh>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/memory_configuration.h>
#include <meta_json_parser/work_group_reader.cuh>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <iostream>
#include <meta_json_parser/action/jarray.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/static_buffer.h>
#include <string>

using namespace boost::mp11;

template<char ...Chars>
using mp_string = mp_list<mp_int<Chars>...>;

int main(int argc, char** argv)
{
    using M3 = MetaMemoryManager<mp_list<void>>;
    using K1 = mp_string<'a', 'b', 'c'>;
    using BA = JDict<mp_list<
        mp_list<K1, JString>
        >>;
    using KW = BA::KeyWriter;
    KW::Buffer b;
    KW::Fill(b);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
