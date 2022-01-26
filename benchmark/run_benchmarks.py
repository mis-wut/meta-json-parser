#!/usr/bin/env python


import pathlib     # file and path handling
import subprocess  # running commands and capturing their output
import re          # parsing output with regular expressions
import csv         # writing results in CSV format

import click       # command line parsing
from tqdm import tqdm, trange # a smart progress meter


def check_exec(exec_path):
	"""TODO: Check '--exec' option for correctness"""
	pass


def check_json_dir(json_dir):
	"""TODO: Check '--json-dir' option for correctness"""
	pass


def time_ns(s):
	"""Convert time in nanoseconds as string to a number

	Parameters
	----------
	s : str
		Time in nanoseconds as string, extracted from the benchmark
		command output.

	Returns
	-------
	int
		Time in nanoseconds as integer value.
	"""
	# int64_t gpu_total = static_cast<int64_t>(ms * 1'000'000.0);
	return int(s, base=10)


@click.command()
@click.option('--exec', '--executable', 'exec_path',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              help="Path to 'meta-json-parser-benchmark' executable",
              default='./meta-json-parser-benchmark')
@click.option('--json-dir',
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Directory with generated JSON files",
              default='../../data/json/generated/')
@click.option('--pattern', 
              help="JSON file name pattern, using {n} placeholder",
              default="sample_{n}.json", show_default=True)
@click.option('--size', '--n_objects', 'size_arg',
              metavar='[NUMBER|scan]',
              help="Number of objects in JSON file to use, or 'scan'",
              default="scan", show_default=True)
@click.option('--output-csv', # uses path_type=click.Path (and not click.File) to support '--append'
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help="Output file in CSV format",
              default="benchmark.csv", show_default=True)
@click.option('--append/--no-append', default=False,
              help="Append to output file (no header)")
@click.option('--samples', metavar='REPETITIONS',
              help='Number of samples (repetitions) with the same values of parameters',
              type=click.IntRange(min=1),
              default='1', show_default=True)
## These arguments should follow arguments of the benchmark executable
@click.option('--use-libcudf-parser', '--use-libcudf', is_flag=True,
              help='Use libcudf JSON parser.',
              default=False, show_default=True)
@click.option('--ws', '--workspace-size',
              help='Workgroup size.',
              type=click.Choice(['32','16','8','4']), show_choices=True,
              default='32', show_default=True)
@click.option('--const-order',
              help='Assumption of keys in JSON in a constant order',
              type=click.Choice(['0', '1']), show_choices=True,
              default='0', show_default=True)
@click.option('-V', '--version', # NOTE: conflicts with same option for version of script
              metavar='[1|2|3]',
              help='Version of dynamic string parsing.',
              type=click.IntRange(1, 3), # inclusive
              default='1', show_default=True)
@click.option('-s', '--max-string-size', 'str_size',
              metavar='BYTES',
              help='Bytes allocated per dynamic string.  Turns on dynamic strings.',
              type=click.IntRange(min=1))
def main(exec_path, json_dir, pattern, size_arg, output_csv, append,
         ws, const_order, version, str_size,
         samples, use_libcudf_parser):
	### run as script

	click.echo(f"Using '{click.format_filename(exec_path)}' executable")
	click.echo(f"('{exec_path.resolve()}')")
	if not use_libcudf_parser:
		click.echo(f"  --workspace-size={ws}")
		click.echo(f"  --const-order={const_order}")
		if str_size is not None:
			click.echo(f"  --max-string-size={str_size}")
			click.echo(f"  --version={version}")
		else:
			click.echo(f"  --version={version} (ignored without --max-string-size=SIZE)")
	else:
		# if using libcudf parser, most options do not matter are is unused
		click.echo(f"  --use-libcudf-parser (assumes executable build with USE_LIBCUDF=1)")

	if samples > 1:
		click.echo(f"  --samples={samples}")

	check_exec(exec_path)
	click.echo(f"JSON files from '{click.format_filename(json_dir)}' directory")
	check_json_dir(json_dir)
	
	if size_arg.isdigit():
		sizes = [int(size_arg, base=10)]
	elif size_arg == 'scan':
		# TODO: maybe find a better way of adding size 10 to beginning
		sizes = [10]+list(range(100000, 900000+1, 100000))
	else:
		sizes = [10]

	if size_arg != 'scan':
		size = sizes[0]
		json_file = json_dir / pattern.format(n=size)
		click.echo(f"Input file is '{json_file}' with {size} objects")

	# if not json_file.is_file():
	#	 click.echo(f"... is not a file")
	#	 exit
	
	results = []
	exec_path = exec_path.resolve()

	for size in tqdm(sizes, desc='size'):
		json_file = json_dir / pattern.format(n=size)
		exec_args = [
			exec_path, json_file, str(size)
		]
		if use_libcudf_parser:
			exec_args.append(f"--use-libcudf-parser")
		else:
			exec_args.extend([
				f"--workspace-size={ws}",
				f"--const-order={const_order}",
				f"--version={version}"
			])
			if str_size is not None:
				exec_args.append(f"--max-string-size={str_size}")

		result = {
			'json file': json_file.name,
			'file size [bytes]': json_file.stat().st_size,
			'number of objects': size,
		}
		if not use_libcudf_parser:
			result.update({
				# those options/parameters are not printed by meta-json-parser-benchmark
				# and you cannot find them in the command output with parse_run_output()
				'max string size': str_size,
			})

		# DEBUG
		#print(f"exec_args = {exec_args}")

		for _ in trange(samples, desc='samples', leave=None):
			process = subprocess.Popen(
				exec_args,
				stdout=subprocess.PIPE
			)
			lines = process.stdout.read().decode('utf-8').split('\n')

			results.append(parse_run_output(lines, result))

	no_header = False
	if output_csv.exists() and append:
		no_header = True

	print(f"Writing benchmarks results to '{output_csv.name}'")
	with output_csv.open('a' if append else 'w') as csv_file:
		csv_writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()), dialect='unix')

		if not no_header:
			csv_writer.writeheader()

		for result in results:
			csv_writer.writerow(result)


def parse_run_output(lines, result = {}):
	"""Parse the output of `meta-json-parser-benchmark` command

	Parameters
	----------
	lines : list of str
		The `meta-json-parser-benchmark` output, split into individual lines.
	result : dict
		The dictionary to store results into.

	Returns
	-------
	result : dict
		The dictionary, with keys naming extracted data, which are the configuration
		values and benchmark results (the latter stores time in nanoseconds it took
		for specific part of the runtime (or totals, or subtotals)).
	"""
	re_string_handling = re.compile('Using (.*) string copy')
	re_assumptions     = re.compile('Assumptions: (.*)')
	re_workgroup_size  = re.compile('Workgroup size: W([0-9]*)')
	re_initialization  = re.compile('\\+ Initialization:\\s*([0-9.]*) ns')
	re_memory          = re.compile('\\+ Memory allocation and copying:\\s*([0-9.]*) ns')
	re_newlines        = re.compile('\\+ Finding newlines offsets \\(indices\\):\\s*([0-9.]*) ns')
	re_parsing_total   = re.compile('\\+ Parsing total \\(sum of the following\\):\\s*([0-9.]*) ns')
	re_json_processing = re.compile('  - JSON processing:\\s*([0-9.]*) ns')
	re_post_processing = re.compile('  - Post kernel hooks:\\s*([0-9.]*) ns')
	re_copying_output  = re.compile('\\+ Copying output:\\s*([0-9.]*) ns')
	re_to_cudf         = re.compile('\\+ Converting to cuDF format:\\s*([0-9.]*) ns')
	re_gpu_total       = re.compile('^Total time measured by GPU:\\s*([0-9.]*) ns')
	re_cpu_total       = re.compile('^Total time measured by CPU:\\s*([0-9.]*) ns')

	# --use-libcudf-parser
	re_build_input_opt = re.compile('\\+ Building input options:\\s*([0-9.]*) ns')
	re_parsing_json    = re.compile('\\+ Parsing json:\\s*([0-9.]*) ns')

	for line in lines:
		match = re_string_handling.match(line)
		if match:
			result['string handling'] = match.group(1)

		match = re_assumptions.match(line)
		if match:
			result['assumptions'] = match.group(1)

		match = re_workgroup_size.match(line)
		if match:
			result['workgroup size'] = int(match.group(1), base=10)

		match = re_initialization.match(line)
		if match:
			result['Initialization [ns]'] = time_ns(match.group(1))

		match = re_memory.match(line)
		if match:
			result['Memory allocation and copying [ns]'] = time_ns(match.group(1))
		
		match = re_newlines.match(line)
		if match:
			result['Finding newlines offsets [ns]'] = time_ns(match.group(1))

		match = re_parsing_total.match(line)
		if match:
			result['TOTAL Parsing time (JSON+hooks) [ns]'] = time_ns(match.group(1))

		match = re_json_processing.match(line)
		if match:
			result['JSON processing [ns]'] = time_ns(match.group(1))

		match = re_post_processing.match(line)
		if match:
			result['Post kernel hooks [ns]'] = time_ns(match.group(1))

		match = re_copying_output.match(line)
		if match:
			result['Copying output [ns]'] = time_ns(match.group(1))

		match = re_to_cudf.match(line)
		if match:
			result['Converting to cuDF format [ns]'] = time_ns(match.group(1))

		# --use-libcudf-parser
		match = re_build_input_opt.match(line)
		if match:
			result['Building input options [ns]'] = time_ns(match.group(1))

		match = re_parsing_json.match(line)
		if match:
			result['Parsing json with libcudf [ns]'] = time_ns(match.group(1))

		# back to generic results
		match = re_gpu_total.match(line)
		if match:
			result['TOTAL time measured by GPU [ns]'] = time_ns(match.group(1))

		match = re_cpu_total.match(line)
		if match:
			result['TOTAL time measured by CPU [ns]'] = time_ns(match.group(1))

	return result


if __name__ == '__main__':
	main()