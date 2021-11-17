#!/bin/bash

# Script used to generate parser configuration of meta-json-parser
# for a given JSON file based on a sample from it (currently it is
# based on the first line of this JSON file), making it possible to
# run the benchmark for this file.
#
# It is meant for testing, in particular finding the time it takes
# to generate benchmark/data_def.cuh, then recompile meta-json-parser-benchmark
#
# Usage:
# $ export JSON2META_DIR=/path/to/json2meta
# $ export METAJSONPARSER_DIR=/path/to/meta-json-parser
# $ ./generate_and_compile.sh path/to/file.json


JSON2META_DIR="${JSON2META_DIR:-$HOME/GPU-IDUB/json2meta}"
JSON2CUDA="$JSON2META_DIR/poc/json2cuda.py"

METAJSONPARSER_DIR="${METAJSONPARSER_DIR:-$HOME/GPU-IDUB/meta-json-parser}"
METAJSONPARSER_BUILDDIR="$METAJSONPARSER_DIR/build/"
METAJSONPARSER_METAHEADER="$METAJSONPARSER_DIR/benchmark/data_def.cuh"

# check configuration
if [ ! -d "$JSON2META_DIR" ]; then
  echo "$JSON2META_DIR directory does not exist (\$JSON2META_DIR)"
  exit 3
fi
if [ ! -d "$METAJSONPARSER_DIR" ]; then
  echo "$METAJSONPARSER_DIR directory does not exist (\$METAJSONPARSER_DIR)"
  exit 3
fi
if [ ! -d "$METAJSONPARSER_BUILDDIR" ]; then
  echo "$METAJSONPARSER_BUILDDIR directory does not exist (\$METAJSONPARSER_BUILDDIR)"
  exit 3
fi

# arguments handling
if [ -z "$1" ]; then
  echo "Usage: $0 JSON_FILE"
  exit 1
fi
if [ ! -r "$1" ]; then
  echo "File '$1' does not exist or is not readable"
  exit 2
fi

# generate sample in a format understood by json2cuda.py script
TMP_FNAME=`mktemp`
head -1 "$1" >"$TMP_FNAME"

# print some debugging information
cat "$TMP_FNAME"

# generate parser configuration
python3 "$JSON2CUDA" "$TMP_FNAME" >"$METAJSONPARSER_METAHEADER"
# compile
make -C "$METAJSONPARSER_BUILDDIR" meta-json-parser-benchmark

# cleanup
rm "$TMP_FNAME"
