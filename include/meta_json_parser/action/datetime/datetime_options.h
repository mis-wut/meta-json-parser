#ifndef META_JSON_PARSER_DATETIME_OPTIONS_H
#define META_JSON_PARSER_DATETIME_OPTIONS_H
#include <boost/mp11.hpp>

struct JDatetimeOptions {
    struct JDatetimeTransformer {
        struct DefaultDatetimeTransformer {
            template<class T>
            inline __device__ T operator()(T c) const { return c; }
        };
    };

    struct TimestampResolution {
        struct Seconds{};
        struct Milliseconds{};
        using Default = Seconds;
    };

private:
    template<class OptionsT>
    using _impl_GetDatetimeTransformer = boost::mp11::mp_map_find<OptionsT, JDatetimeOptions::JDatetimeTransformer>;

    template<class OptionsT>
    using _impl_GetTimestampResolution = boost::mp11::mp_map_find<OptionsT, JDatetimeOptions::TimestampResolution>;
public:
    template<class OptionsT>
    using GetDatetimeTransformer = boost::mp11::mp_eval_if<
    boost::mp11::mp_same<
        _impl_GetDatetimeTransformer<OptionsT>,
        void
    >,
    JDatetimeOptions::JDatetimeTransformer::DefaultDatetimeTransformer,
    boost::mp11::mp_second,
    _impl_GetDatetimeTransformer<OptionsT>
    >;

    template<class OptionsT>
    using GetTimestampResolution = boost::mp11::mp_eval_if<
    boost::mp11::mp_same<
        _impl_GetTimestampResolution<OptionsT>,
        void
    >,
    JDatetimeOptions::TimestampResolution::Default,
    boost::mp11::mp_second,
    _impl_GetTimestampResolution<OptionsT>
    >;
};


#endif //META_JSON_PARSER_DATETIME_OPTIONS_H
