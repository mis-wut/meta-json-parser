#ifndef META_JSON_PARSER_IS_BASE_OF_TEMPLATE_H
#define META_JSON_PARSER_IS_BASE_OF_TEMPLATE_H
#include <utility>
#include <type_traits>

//Source: https://stackoverflow.com/a/34672753

template < template <typename...> class base,typename derived>
struct is_base_of_template_impl
{
    template<typename... Ts>
    static constexpr std::true_type  test(const base<Ts...> *);
    static constexpr std::false_type test(...);
    using type = decltype(test(std::declval<derived*>()));
};

template < template <typename...> class base,typename derived>
using is_base_of_template = typename is_base_of_template_impl<base,derived>::type;

#endif //META_JSON_PARSER_IS_BASE_OF_TEMPLATE_H
