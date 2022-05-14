#!/bin/bash

# default is to use the largest possible file
# from ../../data/json/generated/sample_*.json
#SIZE=1000000

BENCHMARK_DIR="$(dirname "$(realpath -s "$0")")"
BUILD_DIR="$BENCHMARK_DIR/../build"

RUN_BENCHMARKS="$BENCHMARK_DIR/run_benchmarks.py"
BENCHMARK_BIN="$BUILD_DIR/meta-json-parser-benchmark"
#RUN_BENCHMARKS=echo
#BENCHMARK_BIN=echo


run_benchmarks () {
	"$RUN_BENCHMARKS" "$@"
}

run_parser () {
	"$BENCHMARK_BIN" "$@"
}

benchmark () {
	WS=32
	CO=1
	SAMPLES=20
	TYPE="$1"
	shift
	run_benchmarks \
		--const-order=$CO \
		--workspace-size=$WS \
		--pattern="${TYPE}_{n}.json" \
		--size=$SIZE \
		--samples=$SAMPLES \
		--append "$@"
}

check () {
	WS=32
	CO=1
	VER=1
	run_parser \
		--const-order=$CO \
		--workspace-size=$WS \
		--version=$VER \
		--error-checking \
		"$@"
}

build () (
	data_def_file="examples/$1.data_def.cuh"
	data_def_file="$1"

	cmp -s "$data_def_file" "$BENCHMARK_DIR/data_def.cuh" || 
	cp -v "$data_def_file" "$BENCHMARK_DIR/data_def.cuh" &&
	cd "$BUILD_DIR" &&
	make meta-json-parser-benchmark || {
		echo "ERROR building meta-parser"
		exit 3
	}
)

usage () {
	echo "USAGE: $(basename "$0") <data_def file> <json file>"
}

# build and run ./meta-json-parser-benchmark for a given dataset example
build_and_test () {
	if [ $# -lt 2 ]; then
		usage
		echo "At least 2 parameters required, $# provided"
		exit 1
	fi

	type_def="$1"
	jsonfile="$2"

	case "$type_def" in
		*.data_def.cuh)
			;;
		*)
			usage
			echo "Running with anything but *.data_def.cuh file as 1st argument not supported yet"
			echo "argument: '${type_def}'"
			exit 1
			;;
	esac

	case "$jsonfile" in
		*.json | *.jsonl)
			;;
		*)
			usage
			echo "Running with anything but JSON file as 2nd argument is not supported yet"
			echo "argument: '${jsonfile}'"
			exit 1
			;;
	esac

	# sanity check
	if [ ! -f "$jsonfile" ]; then
		usage
		echo "Could not find JSON file '${jsonfile}'"
		exit 2
	fi
	if [ ! -f "${type_def}" ]; then
		usage
		echo "Could not find '${type_def}' file with meta-parser configuration"
		exit 2
	fi

	json_size=$(wc -l <"$jsonfile")
	type_name="$(basename "$type_def" .data_def.cuh)"

	# DEBUG
	echo "jsonfile=$jsonfile"
	echo "type_def=$type_def"
	echo "type_name=$type_name"
	echo "BENCHMARK_DIR=$BENCHMARK_DIR"
	echo "BENCHMARK_BIN=$BENCHMARK_BIN"
	echo "BUILD_DIR=$BUILD_DIR"

	build "$type_def"
	check --output="${type_name}-meta_self.csv" "$jsonfile" $json_size 
	check --output="${type_name}-meta_cudf.csv" --use-libcudf-writer "$jsonfile" $json_size
	check --output="${type_name}-libcudf.csv"   --use-libcudf-parser "$jsonfile" $json_size
	# this one may fail
	check --output="${type_name}-libcudf_dtypes.csv" --use-libcudf-parser --use-dtypes "$jsonfile" $json_size ||
		echo "No --use-dtypes support"

	if [[ -f "${type_name}-meta_cudf.csv" && -f "${type_name}-libcudf.csv" ]]; then
		if ! cmp -b "${type_name}-meta_cudf.csv" "${type_name}-libcudf.csv"; then
			echo "DIFFERENCES for $type_name!"
			head -5 "${type_name}-meta_self.csv" "${type_name}-meta_cudf.csv" "${type_name}-libcudf.csv"
			#echo 
			#diff -u "${type_name}-meta_cudf.csv" "${type_name}-libcudf.csv"
		else
			echo "OK"
		fi
	else
		echo "One or more expected output files are missing:"
		echo "- '${type_name}-meta_cudf.csv'"
		echo "- '${type_name}-libcudf.csv'"
	fi
}

build_and_test "$@"
