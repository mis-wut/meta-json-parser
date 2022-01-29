#ifndef META_JSON_PARSER_OPERATORS_CUH
#define META_JSON_PARSER_OPERATORS_CUH
#include <meta_json_parser/parsing_error.h>

struct NoError {
    typedef bool result_type;
    typedef ParsingError argument_type;

    __host__ __device__ bool operator()(const ParsingError& err)
    {
        return err == ParsingError::None;
    }
};

#endif //META_JSON_PARSER_OPERATORS_CUH
