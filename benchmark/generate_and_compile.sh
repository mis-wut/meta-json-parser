#!/bin/bash

TMP_FNAME=`mktemp`

head -1 $1 > $TMP_FNAME

cat $TMP_FNAME

python3 ~/GPU-IDUB/json2meta/poc/json2cuda.py $TMP_FNAME > ~/GPU-IDUB/meta-json-parser/benchmark/data_def.cuh

make -C ~/GPU-IDUB/meta-json-parser/build/ meta-json-parser-benchmark

rm $TMP_FNAME
