#include <gtest/gtest.h>
#include <thrust/logical.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jarray.cuh>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/meta_utility/map_utility.h>
#include <meta_json_parser/meta_utility/metastring.h>
#include "test_helper.h"

using namespace boost::mp11;

template<class T>
class ExampleJsons : public ::testing::Test { };

TYPED_TEST_SUITE_P(ExampleJsons);

struct no_error {
    typedef bool result_type;
    typedef ParsingError argument_type;

    __host__ __device__ bool operator()(const ParsingError& err)
    {
        return err == ParsingError::None;
    }
};

TYPED_TEST_P(ExampleJsons, Book) {
    using WorkGroupSize = mp_at_c<TypeParam, 0>;
    using ArraySkipping = mp_at_c<TypeParam, 1>;
    using DictOpts = mp_at_c<TypeParam, 2>;
    using DictOrder = mp_at_c<DictOpts, 0>;
    using DictSkipping = mp_at_c<DictOpts, 1>;
    const size_t INPUT_T = 0x8001;
    constexpr int GROUP_SIZE = WorkGroupSize::value;
    constexpr int GROUP_COUNT = 1024 / GROUP_SIZE;
    using GroupCount = mp_int<GROUP_COUNT>;
    using WGR = WorkGroupReader<WorkGroupSize>;
    using RT = RuntimeConfiguration<WorkGroupSize, GroupCount>;
    //Keys
    using K_books = metastring("books");
    using K_isbn = metastring("isbn");
    using K_title = metastring("title");
    using K_subtitle = metastring("subtitle");
    using K_author = metastring("author");
    using K_published = metastring("published");
    using K_publisher = metastring("publisher");
    using K_pages = metastring("pages");
    using K_description = metastring("description");
    using K_website = metastring("website");
    //Bytes to copy
    constexpr int B_isbn        = 16;
    constexpr int B_title       = 128;
    constexpr int B_subtitle    = 128;
    constexpr int B_author      = 64;
    constexpr int B_published   = 64;
    constexpr int B_publisher   = 64;
    constexpr int B_description = 512;
    constexpr int B_website     = 64;
    using A_book_entry = JDict<DictEntries<
            K_isbn,        JStringStaticCopy<mp_int<B_isbn       >, K_isbn       >,
            K_title,       JStringStaticCopy<mp_int<B_title      >, K_title      >,
            K_subtitle,    JStringStaticCopy<mp_int<B_subtitle   >, K_subtitle   >,
            K_author,      JStringStaticCopy<mp_int<B_author     >, K_author     >,
            K_published,   JStringStaticCopy<mp_int<B_published  >, K_published  >,
            K_publisher,   JStringStaticCopy<mp_int<B_publisher  >, K_publisher  >,
            K_pages,       JNumber          <uint32_t,              K_pages      >,
            K_description, JStringStaticCopy<mp_int<B_description>, K_description>,
            K_website,     JStringStaticCopy<mp_int<B_website    >, K_website    >
        >,
        MapEntries<
            JDictOpts::Order, DictOrder,
            JDictOpts::Skip, DictSkipping
        >
    >;
    using A_book_array = JArray<ArrayEntries<
            mp_int<0>, A_book_entry
        >,
        MapEntries<
            JArrayOptions::Skip, ArraySkipping
        >
    >;
    using BA = JDict<DictEntries<
            K_books, A_book_array
        >,
        MapEntries<
            JDictOpts::Order, DictOrder,
            JDictOpts::Skip, DictSkipping
        >
    >;
    using PC = ParserConfiguration<RT, BA>;
    using PK = ParserKernel<PC>;
    const char* json_format =
        "{\n"
        "\"books\": [\n"
        "{\n"
        "\"isbn\": \"%s\",\n"
        "\"title\": \"%s\",\n"
        "\"subtitle\": \"%s\",\n"
        "\"author\": \"%s\",\n"
        "\"published\": \"%s\",\n"
        "\"publisher\": \"%s\",\n"
        "\"pages\": %u,\n"
        "\"description\": \"%s\",\n"
        "\"website\": \"%s\"\n"
        "}]}";
    const char* json_isbn = "9781593275846";
    const char* json_title = "Eloquent JavaScript, Second Edition";
    const char* json_subtitle = "A Modern Introduction to Programming";
    const char* json_author = "Marijn Haverbeke";
    const char* json_published = "2014-12-14T00:00:00.000Z";
    const char* json_publisher = "No Starch Press";
    const char* json_description =
        "JavaScript lies at the heart of almost every modern\\n"
        "web application, from social apps to the newest browser-based games.\\n"
        "Though simple for beginners to pick up and play with, JavaScript is a\\n"
        "flexible, complex language that you can use to build full-scale\\n"
        "applications.";
    const uint32_t json_pages = 472;
    const char* json_website = "http://eloquentjavascript.net/";
    //input
    const int MAX_JSON_LEN = 1024;
    thrust::host_vector<char> h_input(INPUT_T * MAX_JSON_LEN);
    thrust::host_vector<InputIndex> h_indices(INPUT_T + 1);
    auto inp_it = h_input.data();
    auto ind_it = h_indices.begin();
    *ind_it = 0;
    ++ind_it;
    for (int i = 0; i < INPUT_T; ++i)
    {
        inp_it += snprintf(inp_it, MAX_JSON_LEN, json_format,
            json_isbn, json_title, json_subtitle, json_author,
            json_published, json_publisher, json_pages,
            json_description, json_website
        );
        *ind_it = (inp_it - h_input.data());
        ++ind_it;
    }
    thrust::device_vector<char> d_input(h_input);
    thrust::device_vector<InputIndex> d_indices(h_indices);
    using M3 = typename PK::M3;
    using BUF = typename M3::ReadOnlyBuffer;
    thrust::host_vector<BUF> h_buff(1);
    M3::FillReadOnlyBuffer(h_buff[0], nullptr);
    const unsigned int BLOCKS_COUNT = (INPUT_T + GROUP_COUNT - 1) / GROUP_COUNT;
    //correct values
    thrust::host_vector<char> h_c_isbn       (INPUT_T * B_isbn,        '\0');
    thrust::host_vector<char> h_c_title      (INPUT_T * B_title,       '\0');
    thrust::host_vector<char> h_c_subtitle   (INPUT_T * B_subtitle,    '\0');
    thrust::host_vector<char> h_c_author     (INPUT_T * B_author,      '\0');
    thrust::host_vector<char> h_c_published  (INPUT_T * B_published,   '\0');
    thrust::host_vector<char> h_c_publisher  (INPUT_T * B_publisher,   '\0');
    thrust::host_vector<char> h_c_description(INPUT_T * B_description, '\0');
    thrust::host_vector<char> h_c_website    (INPUT_T * B_website,     '\0');
    thrust::host_vector<uint32_t> h_c_pages  (INPUT_T);
    for (int i = 0; i < INPUT_T; i++)
    {
        snprintf(h_c_isbn.data()        + i * B_isbn,        B_isbn,        "%s", json_isbn);
        snprintf(h_c_title.data()       + i * B_title,       B_title,       "%s", json_title);
        snprintf(h_c_subtitle.data()    + i * B_subtitle,    B_subtitle,    "%s", json_subtitle);
        snprintf(h_c_author.data()      + i * B_author,      B_author,      "%s", json_author);
        snprintf(h_c_published.data()   + i * B_published,   B_published,   "%s", json_published);
        snprintf(h_c_publisher.data()   + i * B_publisher,   B_publisher,   "%s", json_publisher);
        snprintf(h_c_description.data() + i * B_description, B_description, "%s", json_description);
        snprintf(h_c_website.data()     + i * B_website,     B_website,     "%s", json_website);
        h_c_pages[i] = json_pages;
    }
    //Readonly buffers
    thrust::device_vector<BUF> d_buff(h_buff);
    //Parsing errors
    thrust::device_vector<ParsingError> d_err(INPUT_T);
    //output values
    thrust::device_vector<char> d_r_isbn       (INPUT_T * B_isbn       );
    thrust::device_vector<char> d_r_title      (INPUT_T * B_title      );
    thrust::device_vector<char> d_r_subtitle   (INPUT_T * B_subtitle   );
    thrust::device_vector<char> d_r_author     (INPUT_T * B_author     );
    thrust::device_vector<char> d_r_published  (INPUT_T * B_published  );
    thrust::device_vector<char> d_r_publisher  (INPUT_T * B_publisher  );
    thrust::device_vector<char> d_r_description(INPUT_T * B_description);
    thrust::device_vector<char> d_r_website    (INPUT_T * B_website    );
    thrust::device_vector<uint32_t> d_r_pages  (INPUT_T);
    thrust::host_vector<void*> h_outputs(9);
    h_outputs[PK::KC::OM::template TagIndex<K_isbn>::value       ] = d_r_isbn.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_title>::value      ] = d_r_title.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_subtitle>::value   ] = d_r_subtitle.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_author>::value     ] = d_r_author.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_published>::value  ] = d_r_published.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_publisher>::value  ] = d_r_publisher.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_description>::value] = d_r_description.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_website>::value    ] = d_r_website.data().get();
    h_outputs[PK::KC::OM::template TagIndex<K_pages>::value      ] = d_r_pages.data().get();
    thrust::device_vector<void*> d_outputs(h_outputs);
    thrust::fill(d_err.begin(), d_err.end(), ParsingError::None);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
    typename PK::Launcher(&_parser_kernel<PC>)(BLOCKS_COUNT)(
        d_buff.data().get(),
        d_input.data().get(),
        d_indices.data().get(),
        d_err.data().get(),
        d_outputs.data().get(),
        INPUT_T
    );
    ASSERT_TRUE(cudaGetLastError() == cudaError::cudaSuccess);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaError::cudaSuccess);
    thrust::host_vector<ParsingError> h_err(d_err);
    thrust::host_vector<char> h_r_isbn(d_r_isbn);
    thrust::host_vector<char> h_r_title(d_r_title);
    thrust::host_vector<char> h_r_subtitle(d_r_subtitle);
    thrust::host_vector<char> h_r_author(d_r_author);
    thrust::host_vector<char> h_r_published(d_r_published);
    thrust::host_vector<char> h_r_publisher(d_r_publisher);
    thrust::host_vector<char> h_r_description(d_r_description);
    thrust::host_vector<char> h_r_website(d_r_website);
    thrust::host_vector<uint32_t> h_r_pages(d_r_pages);
    ASSERT_TRUE(thrust::all_of(d_err.begin(), d_err.end(), no_error()));
    ASSERT_TRUE(thrust::equal(h_r_isbn.begin(), h_r_isbn.end(), h_c_isbn.begin()));
    ASSERT_TRUE(thrust::equal(h_r_title.begin(), h_r_title.end(), h_c_title.begin()));
    ASSERT_TRUE(thrust::equal(h_r_subtitle.begin(), h_r_subtitle.end(), h_c_subtitle.begin()));
    ASSERT_TRUE(thrust::equal(h_r_author.begin(), h_r_author.end(), h_c_author.begin()));
    ASSERT_TRUE(thrust::equal(h_r_published.begin(), h_r_published.end(), h_c_published.begin()));
    ASSERT_TRUE(thrust::equal(h_r_publisher.begin(), h_r_publisher.end(), h_c_publisher.begin()));
    ASSERT_TRUE(thrust::equal(h_r_description.begin(), h_r_description.end(), h_c_description.begin()));
    ASSERT_TRUE(thrust::equal(h_r_website.begin(), h_r_website.end(), h_c_website.begin()));
    ASSERT_TRUE(thrust::equal(h_r_pages.begin(), h_r_pages.end(), h_c_pages.begin()));
}

REGISTER_TYPED_TEST_SUITE_P(ExampleJsons, Book);

using ArraySkippingOption = mp_list<
    JArrayOptions::Skip::Disable,
    JArrayOptions::Skip::Enable_c<8>
>;

using DictOptions = mp_list<
    mp_list<JDictOpts::Order::RandomOrder, JDictOpts::Skip::Disable>,
    mp_list<JDictOpts::Order::RandomOrder, JDictOpts::Skip::Enable_c<8>>,
    mp_list<JDictOpts::Order::ConstOrder, JDictOpts::Skip::Disable>
>;

using AllWorkGroupsWithArrayDictOptsTypes = mp_rename<mp_product<
    mp_list,
    AllWorkGroups,
    ArraySkippingOption,
    DictOptions
>, ::testing::Types>;

struct NameGenerator {
    template <typename TypeParam>
    static std::string GetName(int i) {
        using GroupSize = mp_at_c<TypeParam, 0>;
        using ArraySkipping = mp_at_c<TypeParam, 1>;
        using DictOpts = mp_at_c<TypeParam, 2>;
        using DictOrder = mp_at_c<DictOpts, 0>;
        using DictSkipping = mp_at_c<DictOpts, 1>;

        std::stringstream stream;
        stream << "WS_" << GroupSize::value;
        stream << "_";

        if constexpr (std::is_same_v<ArraySkipping , JArrayOptions::Skip::Disable>) {
            stream << "array_skip_disabled";
        } else {
            stream << "array_skip_enabled";
        }
        stream << "_";

        if constexpr (std::is_same_v<DictSkipping , JDictOpts::Skip::Disable>) {
            stream << "dict_skip_disabled";
        } else {
            stream << "dict_skip_enabled";
        }
        stream << "_";

        if constexpr (std::is_same_v<DictOrder, JDictOpts::Order::ConstOrder>) {
            stream << "const_order";
        } else {
            stream << "random_order";
        }

        return stream.str();
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(AllWorkGroupsWithArrayDictOpts, ExampleJsons, AllWorkGroupsWithArrayDictOptsTypes, NameGenerator);
