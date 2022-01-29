#ifndef META_JSON_PARSER_TEST_CONFIGURATION_H
#define META_JSON_PARSER_TEST_CONFIGURATION_H

#if _DEBUG
constexpr size_t TEST_SIZE = 0x11;
#else
constexpr size_t TEST_SIZE = 0x8001;
#endif

constexpr size_t SEED = 0xDEAD;

#endif //META_JSON_PARSER_TEST_CONFIGURATION_H
