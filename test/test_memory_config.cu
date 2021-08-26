#include <meta_json_parser/memory/pool.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace boost::mp11;
using namespace meta_json::memory;
using namespace meta_json::memory::detail;


namespace {
	struct InitTag {};
}

TEST(MemoryConfigTests, PoolFiltersTest)
{
	using allocations = mp_list<
			allocation<int, segment::constant, InitTag>,
			allocation<int, segment::global, InitTag>,
			allocation<double, segment::shared, void>,
			allocation<float, segment::constant, InitTag>,
			allocation<char, segment::shared>
		>;

	using expect_cinit = mp_list<
			allocation<int, segment::constant, InitTag>,
			allocation<float, segment::constant, InitTag>
		>;
	using expect_ginit = mp_list<
			allocation<int, segment::global, InitTag>
		>;
	using expect_sscratch = mp_list<
			allocation<double, segment::shared, void>,
			allocation<char, segment::shared, void>
		>;
	using cinit = init_allocations<allocations, segment::constant>;
	using cscratch = scratch_allocations<allocations, segment::constant>;

	using ginit = init_allocations<allocations, segment::global>;
	using gscratch = scratch_allocations<allocations, segment::global>;

	using sinit = init_allocations<allocations, segment::shared>;
	using sscratch = scratch_allocations<allocations, segment::shared>;

	using test_cinit = std::is_same<cinit, expect_cinit>;
	using test_cscratch = std::is_same<cscratch, mp_list<>>;

	using test_ginit = std::is_same<ginit, expect_ginit>;
	using test_gscratch = std::is_same<gscratch, mp_list<>>;

	using test_sinit = std::is_same<sinit, mp_list<>>;
	using test_sscratch = std::is_same<sscratch, expect_sscratch>;

	EXPECT_TRUE(test_cinit::value);
	EXPECT_TRUE(test_cscratch::value);
	EXPECT_TRUE(test_ginit::value);
	EXPECT_TRUE(test_gscratch::value);
	EXPECT_TRUE(test_sinit::value);
	EXPECT_TRUE(test_sscratch::value);
}

template<typename T>
using alloc_c = allocation<T, segment::constant, InitTag>;

template<typename T>
using alloc_g = allocation<T, segment::global>;

template<typename T>
using alloc_gi = allocation<T, segment::global, InitTag>;

template<typename T>
using alloc_s = allocation<T, segment::shared>;

template<typename Node, typename... Ts>
using test_init_subtree = mp_and<
	mp_is_set<subtree_init_allocations<Node, segment::constant>>,
	mp_empty<
		mp_set_difference<
			mp_list<alloc_c<Ts>...>,
			subtree_init_allocations<Node, segment::constant>
		>
	>
>;

TEST(MemoryConfigTests, PoolInitFilterTest)
{
	using leaf1 = config_node<
		mp_list<
			alloc_c<float>,
			alloc_c<int>
		>
	>;

	using leaf2 = config_node<
		alloc_c<float>
	>;

	using innode = config_node<
		mp_list<alloc_c<char>, alloc_c<short>>,
		leaf2
	>;

	using root = config_node <
		alloc_c<int>,
		leaf1,
		innode
	>;
	using test_leaf1 = test_init_subtree<leaf1, float, int>;
	using test_leaf2 = test_init_subtree<leaf2, float>;
	using test_innode = test_init_subtree<innode, char, short, float>;
	using test_root = test_init_subtree<root, int, char, short, float>;
	EXPECT_TRUE(test_leaf1::value);
	EXPECT_TRUE(test_leaf2::value);
	EXPECT_TRUE(test_innode::value);
	EXPECT_TRUE(test_root::value);
}

TEST(MemoryConfigTests, PoolLayoutTest)
{
	using pool = mp_list<
		alloc_g<char>,
		alloc_g<float>,
		alloc_g<InitTag>,
		alloc_g<int>,
		alloc_g<double>,
		alloc_g<char>,
		alloc_g<char[3]>,
		alloc_g<int>>;
	using map = buffer_map<pool, mp_size_t<10>>;
	using map_end = buffer_end<pool, mp_size_t<10>>;

	using expected_map = mp_list<
		mp_list<alloc_g<char>, mp_size_t<10>>,
		mp_list<alloc_g<float>, mp_size_t<12>>,
		mp_list<alloc_g<InitTag>, mp_size_t<16>>,
		mp_list<alloc_g<int>, mp_size_t<16>>,
		mp_list<alloc_g<double>, mp_size_t<24>>,
		mp_list<alloc_g<char>, mp_size_t<32>>,
		mp_list<alloc_g<char[3]>, mp_size_t<33>>,
		mp_list<alloc_g<int>, mp_size_t<36>>
	>;
	using expected_end = mp_size_t<40>;

	using test_map = mp_same<map, expected_map>;

	EXPECT_TRUE(test_map::value);
	EXPECT_EQ(map_end::value, expected_end::value);

	using empty_pool = mp_list<>;

	using empty_map = buffer_map<empty_pool, mp_size_t<5>>;
	using empty_end = buffer_end<empty_pool, mp_size_t<5>>;

	using expected_emtpy_map = mp_list<>;
	using expected_empty_end = mp_size_t<5>;

	using test_empty_map = mp_same<empty_map, expected_emtpy_map>;
	
	EXPECT_TRUE(test_empty_map::value);
	EXPECT_EQ(empty_end::value, expected_empty_end::value);
}

namespace
{
	struct TestWrapperBase
	{
		static thread_local std::vector<size_t> constructed;

		TestWrapperBase()
		{
			id = constructed.size();
			constructed.push_back(id);
		}
		size_t id;
	};

	thread_local std::vector<size_t> TestWrapperBase::constructed;

	template<typename T>
	struct alignas(64) TestWrapper : public TestWrapperBase
	{
		using type = T;
		
		TestWrapper()
		{
			value = sizeof(T);
		}
		T value;
	};

	struct TestInitBase
	{
		static thread_local std::vector<intptr_t> pointers;

	};

	thread_local std::vector<intptr_t> TestInitBase::pointers;

	template<typename T>
	struct TestInit : public TestInitBase
	{
		void operator()(T* data)
		{
			SCOPED_TRACE(typeid(TestInit).name());
			EXPECT_EQ(data->value, sizeof(typename T::type));
			data->value += 10;
			pointers.push_back(reinterpret_cast<intptr_t>(data));
		}
	};

	void clear_test_data()
	{
		TestInitBase::pointers.clear();
		TestWrapperBase::constructed.clear();
	}
}

template<typename T>
using testwrapper_alloc = allocation<TestWrapper<T>, segment::global, TestInit<TestWrapper<T>>>;

TEST(MemoryConfigTests, PoolInitalizationTest)
{
	clear_test_data();
	using root = config_node<mp_list<
			testwrapper_alloc<int>,
			testwrapper_alloc<double>,
			testwrapper_alloc<char>
		>>;
	using config = pool<root>;
	intptr_t base_ptr;
	config::initialize_pool<segment::global>([&](std::byte* host_data)
	{
		base_ptr = reinterpret_cast<intptr_t>(host_data);
		TestWrapper<int>* int_wrap = reinterpret_cast<TestWrapper<int>*>(TestInitBase::pointers[0]);
		TestWrapper<double>* double_wrap = reinterpret_cast<TestWrapper<double>*>(TestInitBase::pointers[1]);
		TestWrapper<char>* char_wrap = reinterpret_cast<TestWrapper<char>*>(TestInitBase::pointers[2]);
		//tests need to go here, since after the function exits memory has already been freed and the pointers
		//are inaccessible.
		EXPECT_EQ(int_wrap->value, sizeof(int)+10);
		EXPECT_EQ(double_wrap->value, sizeof(double)+10);
		EXPECT_EQ(char_wrap->value, sizeof(char)+10);
	});
	EXPECT_EQ(config::global_alignment, 64);
	EXPECT_EQ((base_ptr & (config::global_alignment - 1)), 0);
	EXPECT_THAT(TestInitBase::pointers, testing::ElementsAre(base_ptr, base_ptr + 64, base_ptr + 128));
	EXPECT_THAT(TestWrapperBase::constructed, testing::ElementsAre(0, 1, 2));
}

TEST(MemoryConfigTests, AllocatorStateTest)
{
	using empty = mp_list<>;
	using ce = mp_list < segment::constant, empty, mp_size_t<16>>;
	using ge = mp_list<segment::global, empty, mp_size_t<8>>;
	using se = mp_list<segment::shared, empty, mp_size_t<0>>;
	using segm_map = mp_list<ce, ge, se>;

	using first = allocator_state<segm_map>;
	std::byte dummy;
	std::byte* dummy_ptr = &dummy;
	first f{ {dummy_ptr, dummy_ptr, dummy_ptr} };

	EXPECT_EQ(f.scratch_ptr<segment::shared>(), dummy_ptr);
	EXPECT_EQ(f.scratch_ptr<segment::global>(), dummy_ptr + 8);
	EXPECT_EQ(f.scratch_ptr<segment::constant>(), dummy_ptr + 16);

	auto s = f.advance<mp_list<segment::shared, mp_size_t<12>>, mp_list<segment::constant, mp_size_t<4>>>();

	EXPECT_EQ(s.scratch_ptr<segment::shared>(), dummy_ptr + 12);
	EXPECT_EQ(s.scratch_ptr<segment::global>(), dummy_ptr + 8);
	EXPECT_EQ(s.scratch_ptr<segment::constant>(), dummy_ptr + 20);
}