#pragma once
#include <cstdint>
#include <limits>

namespace __ParsingError_impl {
	using ParsingErrorBackendType = uint8_t;
}

/// <summary>
/// Indication of an error occurence during parsing. None = 0 for no error.
/// </summary>
enum class ParsingError : __ParsingError_impl::ParsingErrorBackendType
{
	None = 0,
	ParseNull_Mismatch,
	ParseBoolean_Mismatch,
	ParseInt32_InvalidZero,
	ParseInt32_ZeroLength,
	ParseInt32_InvalidCharacter,
	ParseFloat_InvalidZero,
	ParseFloat_Integral_ZeroLength,
	ParseFloat_Integral_InvalidCharacter,
	ParseFloat_Fraction_ZeroLength,
	ParseFloat_Fraction_InvalidCharacter,
	ParseFloat_Fraction_InvalidCharacter_Skip,
	ParseFloat_Exponent_ZeroLength,
	ParseFloat_Exponent_InvalidCharacter,
	ParseFloat_Exponent_InvalidCharacter_Skip,
	ParseString_NoQuotationBegin,
	ParseString_InvalidCharsEscaped,
	ParseString_InvalidBytes,
	Skip_NotExpectedValueInContainer,
	Skip_MismatchComma,
	Skip_TooDeep_Brace,
	Skip_TooDeep_Braket,
	Skip_TooDeep_KeyValue,
	Skip_NoObjectToEnd,
	Skip_NoArrayToEnd,
	Skip_UnknownCharacter,
	FindNext_UnexpectedByte,
	FindNext_UnexpectedCharacter,
	FindNoneWhite_UnexpectedByte,
	KeyResolver_DoesntStartWithBrace,
	KeyResolver_NoMatchingKey,
	KeyResolver_MissingComma,
	IndexResolver_DoesntStartWithBracket,
	IndexResolver_ExceptedCommaOrBracket,
	Other = std::numeric_limits<__ParsingError_impl::ParsingErrorBackendType>::max()
};

