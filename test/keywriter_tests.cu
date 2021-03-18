#include <gtest/gtest.h>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/mp_string.h>
#include <algorithm>

using namespace boost::mp11;

class KeyWriterTest : public ::testing::Test {
};

using K_abc = mp_string<'a', 'b', 'c'>;
using K_xyz = mp_string<'x', 'y', 'z'>;
using K_co = mp_string<'c', 'o'>;
using K_V = mp_string<'V'>;
using K_12345 = mp_string<'1', '2', '3', '4', '5'>;
using K_longestkey = mp_string<'l', 'o', 'n', 'g', 'e', 's', 't', 'k', 'e', 'y'>;

template<class DictT>
void DictCheck(char (&correct)[DictT::KeyWriter::StorageSize::value + 1])
{
	using KW = typename DictT::KeyWriter;
	typename KW::Buffer b;
	KW::Fill(b);
	auto& a = b.template Alias<char[sizeof(b)]>();
	std::vector<char> fixed(sizeof(b));
	std::transform(&correct[0], &correct[sizeof(b)], fixed.begin(),
		[](char c) {
		return c == '_' ? '\0' : c;
	});
	ASSERT_TRUE(std::equal(fixed.begin(), fixed.end(), &a[0]));
}

TEST_F(KeyWriterTest, single_key) {
	using Dict = JDict<mp_list<
		mp_list<K_abc, JString>
		>>;
	char correct[] = "___a___b___c";
	DictCheck<Dict>(correct);
}

TEST_F(KeyWriterTest, four_short_keys) {
	using Dict = JDict<mp_list<
		mp_list<K_xyz, JString>,
		mp_list<K_co, JString>,
		mp_list<K_abc, JString>,
		mp_list<K_V, JString>
		>>;
	char correct[] = "Vacx_boy_c_z";
	DictCheck<Dict>(correct);
}

TEST_F(KeyWriterTest, four_long_keys) {
	using Dict = JDict<mp_list<
		mp_list<K_xyz, JString>,
		mp_list<K_12345, JString>,
		mp_list<K_longestkey, JString>,
		mp_list<K_V, JString>
		>>;
	char correct[] =
		"Vl1x_o2y_n3z_g4__e5__s___t___k___e___y__";
	DictCheck<Dict>(correct);
}


TEST_F(KeyWriterTest, six_keys) {
	using Dict = JDict<mp_list<
		mp_list<K_xyz, JString>,
		mp_list<K_12345, JString>,
		mp_list<K_longestkey, JString>,
		mp_list<K_V, JString>,
		mp_list<K_co, JString>,
		mp_list<K_abc, JString>
		>>;
	char correct[] =
		"Vl1x_o2y_n3z_g4__e5__s___t___k___e___y__"
		"__ac__bo__c_____________________________";
	DictCheck<Dict>(correct);
}
