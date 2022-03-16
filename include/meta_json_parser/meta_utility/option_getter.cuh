#ifndef META_JSON_PARSER_OPTION_GETTER_CUH
#define META_JSON_PARSER_OPTION_GETTER_CUH

#define OPTION_GETTER(OPT_FAMILY, OPT)\
private:\
    template<class OptionsT>\
    using _impl_Get##OPT = boost::mp11::mp_map_find<OptionsT, OPT_FAMILY :: OPT >;\
public:\
    template<class OptionsT>\
    using Get##OPT = boost::mp11::mp_eval_if<\
        boost::mp11::mp_same<\
            _impl_Get##OPT <OptionsT>,\
            void\
        >,\
        OPT_FAMILY :: OPT ::Default,\
        boost::mp11::mp_second,\
        _impl_Get##OPT <OptionsT>\
    >;

#endif //META_JSON_PARSER_OPTION_GETTER_CUH
