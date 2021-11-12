#!/bin/bash


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
python3 ~/GPU-IDUB/json2meta/poc/json2cuda.py "$TMP_FNAME" >~/GPU-IDUB/meta-json-parser/benchmark/data_def.cuh

# compile
make -C ~/GPU-IDUB/meta-json-parser/build/ meta-json-parser-benchmark

# cleanup
rm "$TMP_FNAME"
