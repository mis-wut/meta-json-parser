#pragma once
#include <boost/mp11/integral.hpp>
#include <boost/mp11/list.hpp>

namespace boost {
namespace mp11 {
	template<char ...Chars>
	using mp_string = mp_list<mp_int<Chars>...>;
}
}
