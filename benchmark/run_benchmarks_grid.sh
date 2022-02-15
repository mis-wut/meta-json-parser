#!/bin/bash

# This script is used to benchmark all possible combinations
# of specifics of parsing with meta-json-parser ("grid search").
# It uses run_benchmarks.py script for each set of parameters

# default is to use the largest possible file
# from ../../data/json/generated/sample_*.json
SIZE=900000

RUN_BENCHMARKS="$(dirname $0)/run_benchmarks.py"
#RUN_BENCHMARKS=echo

string_handling () {
	"$RUN_BENCHMARKS" "$@"

	"$RUN_BENCHMARKS" --max-string-size=32 --version=1 "$@"
	"$RUN_BENCHMARKS" --max-string-size=32 --version=2 "$@"
	"$RUN_BENCHMARKS" --max-string-size=32 --version=3 "$@"
}

run () {
	for WS in 32 16 8 4; do
		for CO in 0 1; do
			string_handling \
				--const-order=$CO \
				--workspace-size=$WS \
				--size=$SIZE --append "$@"
		done
	done
}

run "$@"
