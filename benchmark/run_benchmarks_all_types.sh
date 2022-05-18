#!/bin/bash

# default is to use the largest possible file
# from ../../data/json/generated/sample_*.json
#SIZE=1000000

BENCHMARK_DIR="$(dirname "$(realpath -s "$0")")"
EXAMPLES_DIR="$BENCHMARK_DIR/examples"
EXAMPLES_DIR="$(realpath -s "$BENCHMARK_DIR/../../data/json/generated_types_bench")"
NOTEBOOKS_DIR="$BENCHMARK_DIR/notebooks"
BUILD_DIR="$BENCHMARK_DIR/../build"

RUN_CHECK="$BENCHMARK_DIR/run_benchmarks_types.sh"
RUN_BENCHMARKS="$BENCHMARK_DIR/run_benchmarks.py"
RUN_GENERATE="$BENCHMARK_DIR/generate_types_bench.py"
BENCHMARK_BIN="$BUILD_DIR/meta-json-parser-benchmark"


benchmark () {
	typename="$1"
	size=1000000
	shift

	WS=32
	CO=1
	SAMPLES=50

	"$RUN_BENCHMARKS" \
		--const-order=$CO \
		--workspace-size=$WS \
		--json-dir="$EXAMPLES_DIR" \
		--pattern="${typename}_{n}.jsonl" \
		--size=$size \
		--samples=$SAMPLES \
		"$@"
}
run_generate () {
	"$RUN_GENERATE" "$@"
}

run_benchmarks () {
	typename="${1}1a"
	size=1000000

	# DEBUG
	#echo "typename=$typename; size=$size"

	if [ ! -f "$EXAMPLES_DIR/${typename}_10.jsonl" ]; then
		head -10 \
			 "$EXAMPLES_DIR/${typename}_${size}.jsonl" \
			>"$EXAMPLES_DIR/${typename}_10.jsonl"
	fi

	"$RUN_CHECK" "$EXAMPLES_DIR/${typename}.data_def.cuh" "$EXAMPLES_DIR/${typename}_10.jsonl"
	benchmark "${typename}" \
		--output-csv="$NOTEBOOKS_DIR/benchmark_metaparser-co=1_cudf=21.10-docker_${typename}.csv"
	benchmark "${typename}" \
		--use-libcudf-parser \
		--output-csv="$NOTEBOOKS_DIR/benchmark_libcudf_cudf=21.10-docker_${typename}.csv"
	benchmark "${typename}" \
		--use-libcudf-parser --use-dtypes \
		--output-csv="$NOTEBOOKS_DIR/benchmark_libcudf_cudf=21.10-docker_${typename}_dtypes.csv"
}


run_generate_all () {
	size=1000000

	run_generate --json-dir="$EXAMPLES_DIR" --type=datetime
	run_generate --json-dir="$EXAMPLES_DIR" --type=bool
	run_generate --json-dir="$EXAMPLES_DIR" --type=string
	run_generate --json-dir="$EXAMPLES_DIR" --type=nullable_string
	run_generate --json-dir="$EXAMPLES_DIR" --type=integer
	run_generate --json-dir="$EXAMPLES_DIR" --type=fixed
	run_generate --json-dir="$EXAMPLES_DIR" --type=float
}

run_benchmarks_all () {
	#for typename in 'datetime' 'bool' 'string' 'nullable_string' 'integer' 'fixed' 'float'; do
	for typename in 'datetime' 'string' 'nullable_string' 'integer' 'fixed'; do
		run_benchmarks "$typename"
	done
}


# NOTE: needs pandas
#run_generate_all

# NOTE: needs meta-json-parser-benchmark
run_benchmarks_all
#run_benchmarks datetime
#run_benchmarks bool
#run_benchmarks string
#run_benchmarks nullable_string
#run_benchmarks integer
#run_benchmarks fixed
#run_benchmarks float

#"$RUN_CHECK" "$EXAMPLES_DIR/bool.data_def.cuh" "$EXAMPLES_DIR/bool_10.json"
#"$RUN_BENCHMARKS" --workspace-size=32 --const-order=1 --samples=50 \
#  --json-dir="$EXAMPLES_DIR" --size=1000000 --pattern='bool_1000000.json' \
#  --output-csv="$NOTEBOOKS_DIR/benchmark_metaparser_cudf=21.10-docker_bool.csv"
#"$RUN_BENCHMARKS"