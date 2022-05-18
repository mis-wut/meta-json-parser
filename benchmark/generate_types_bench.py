#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
import textwrap
import sys
import pathlib

import click


def gen_sample_and_data_def(size, typename):
	rng = np.random.default_rng()

	data = {}
	desc = ""
	# TODO: remove code duplication
	if typename == "datetime":
		# "D" is calendar day frequency
		# "S" is seconds frequency
		data = pd.date_range(
			start='2010-01-01', end='2021-01-01',
            periods=size
		).round("S").strftime("%Y.%m.%d %H:%M:%S")
		desc = """\
		// INCLUDES
		#include <meta_json_parser/action/datetime/jdatetime.cuh>
		#include <meta_json_parser/meta_utility/metastring.h>

		// EXAMPLE:
		// {"a":"2021-03-18 00:16:48"}

		// KEYS
		using K_L1_a = mp_string<'a'>;

		// FORMATS
		using DatetimeFormat = metastring("%Y-%m-%d %H:%M:%S");

		// DICT
		#define STATIC_STRING_SIZE 32
		template<template<class, int> class StringFun, class DictOpts>
		using DictCreator = JDict < mp_list <
			mp_list<K_L1_a, JDatetime<DatetimeFormat, int64_t, K_L1_a>>,
		>,
			DictOpts
		> ;

		// DTYPES
		#ifdef HAVE_LIBCUDF
		#define HAVE_DTYPES
		std::map< std::string, cudf::data_type > dtypes{
			{ "a", cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS } }
		};
		#endif
		"""
	
	elif typename == "bool":
		data = rng.choice([True, False], size=size)
		desc = """\
		// INCLUDES
		#include <meta_json_parser/action/jbool.cuh>

		// EXAMPLE:
		// {"a":false}

		// KEYS
		using K_L1_a = mp_string<'a'>;

		// DICT
		#define STATIC_STRING_SIZE 32
		template<template<class, int> class StringFun, class DictOpts>
		using DictCreator = JDict < mp_list <
		    mp_list<K_L1_a, JBool<uint8_t, K_L1_a>> 
		>,
		    DictOpts
		> ;

		#ifdef HAVE_LIBCUDF
		#define HAVE_DTYPES
		std::map< std::string, cudf::data_type > dtypes{
			{ "a", cudf::data_type{cudf::type_id::BOOL8} },
		};
		#endif
		"""

	elif typename in {"string", "nullable_string"}:
		data = pd.Series(rng.integers(low=0, high=1000, size=size))\
			.map(lambda x: f'name_{x}')
		if typename == "nullable_string":
			typefmt = "NullDefaultEmptyString<JStringVariant<K_L1_a, STATIC_STRING_SIZE>>"
		else:
			typefmt = "StringFun<K_L1_a, STATIC_STRING_SIZE>"
		desc = f"""\
		// INCLUDES
		#include <meta_json_parser/action/jstring.cuh>
		#include <meta_json_parser/action/decorators/null_default_value.cuh>
		
		// EXAMPLE
		// {{"a":"name_376"}}

		// KEYS
		using K_L1_a = mp_string<'a'>;

		// CONFIGURATION
		template<class Key, int Size, class Options = boost::mp11::mp_list<>>
		using JStringVariant = JStringStaticCopy<mp_int<Size>, Key, Options>;

		// DICT
		#define STATIC_STRING_SIZE 32
		template<template<class, int> class StringFun, class DictOpts>
		using DictCreator = JDict < mp_list <
		    mp_list<K_L1_a, {typefmt}>,
		>,
		    DictOpts
		> ;

		"""

		if typefmt == "nullable_string":
			desc += '#pragma message("Always using JStringStaticCopy for parsing strings")\n\n'
		
		desc += """\
		#ifdef HAVE_LIBCUDF
		#define HAVE_DTYPES
		std::map< std::string, cudf::data_type > dtypes{
			{ "a", cudf::data_type{cudf::type_id::STRING} },
		};
		#endif
		"""

	elif typename == "integer":
		data = rng.integers(low=0, high=9000, size=size)
		desc = """\
		// INCLUDES
		#include <meta_json_parser/action/jnumber.cuh>

		// EXAMPLE
		// {"a":163}

		// KEYS
		using K_L1_a = mp_string<'a'>;

		// DICT
		#define STATIC_STRING_SIZE 32
		template<template<class, int> class StringFun, class DictOpts>
		using DictCreator = JDict < mp_list <
		    mp_list<K_L1_a, JNumber<int32_t, K_L1_a>>,
		>,
		    DictOpts
		> ;

		
		#ifdef HAVE_LIBCUDF
		#define HAVE_DTYPES
		std::map< std::string, cudf::data_type > dtypes{
			{ "a", cudf::data_type{cudf::type_id::INT32} },
		};
		#endif
		"""

	elif typename in {"fixed", "float"}:
		data = pd.Series(rng.uniform(low=-180.0, high=180.0, size=size))\
			.map(lambda x: float(f'{x:.3f}'))
		if typename == "fixed":
			typefmt = "JRealNumber<float, K_L1_a>"
		else:
			typefmt = "JRealNumber<float, K_L1_a, JRealOptionsFixedFormat>"
		desc = f"""\
		// INCLUDES
		#include <meta_json_parser/action/jrealnumber.cuh>

		// EXAMPLE
		// {{"a":47.592}}

		// KEYS
		using K_L1_a = mp_string<'a'>;

		// OPTIONS
		using JRealOptionsFixedFormat = boost::mp11::mp_list<
		    boost::mp11::mp_list<
		        JRealNumberOptions::JRealNumberExponent,
		        JRealNumberOptions::JRealNumberExponent::WithoutExponent
		    >
		>;

		// DICT
		#define STATIC_STRING_SIZE 32
		template<template<class, int> class StringFun, class DictOpts>
		using DictCreator = JDict < mp_list <
		    mp_list<K_L1_a, {typefmt}>,
		>,
		    DictOpts
		> ;

		
		#ifdef HAVE_LIBCUDF
		#define HAVE_DTYPES
		std::map< std::string, cudf::data_type > dtypes{{
			{{ "a", cudf::data_type{{cudf::type_id::FLOAT32}} }},
		}};
		#endif
		"""

	else:
		print(f"{typename} type not supported, exiting", file=stderr)
		sys.exit(1)


	# DEBUG
	#print(f"data={data}")
	return (
		pd.DataFrame({"a": data}),
		textwrap.dedent(desc)
	)


@click.command()
@click.option('--json-dir',
              type=click.Path(exists=True,
                              file_okay=False, path_type=pathlib.Path),
              help="Directory with generated JSON files",
              default='../../data/json/generated_types_bench/')
@click.option('--size', '--n_objects',
              metavar='NUMBER',
              help="Number of objects in JSON file to generate.",
			  type=click.IntRange(min=1),
              default=1000000, show_default=True)
@click.option('--type', 'typename',
              help='Element type in JSON object',
              type=click.Choice([
                  'datetime', 'bool', 'string', 'nullable_string', 'integer', 'fixed', 'float'
              ]),
              show_choices=True,
              default='datetime', show_default=True)
def main(json_dir, size, typename):
	click.echo(f"Generating {size} objects in JSON")
	(df, hdr) = gen_sample_and_data_def(size, typename)

	click.echo(f"Writing results to '{click.format_filename(json_dir)}' directory")
	basename=pathlib.Path(json_dir)\
		.joinpath(typename + '1a' + '_' + str(size))
	jsonpath=basename.with_suffix('.jsonl')
	click.echo(f"Writing JSONL to '{click.format_filename(jsonpath)}' file") 
	df.to_json(jsonpath, orient='records', lines=True, date_format='iso')

	basename=pathlib.Path(json_dir)\
		.joinpath(typename + '1a')
	cuhpath=basename.with_suffix('.data_def.cuh')
	click.echo(f"Writing Meta Parser config to '{click.format_filename(cuhpath)}' file")
	cuhpath.write_text(hdr)


if __name__ == "__main__":
    main()