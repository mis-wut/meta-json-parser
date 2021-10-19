#!/usr/bin/env python


import pathlib     # file and path handling
import subprocess  # running commands and capturing their output
import re          # parsing output with regular expressions
import csv         # writing results in CSV format

import click       # command line parsing


def check_exec(exec_path):
	"""TODO: Check '--exec' option for correctness"""
	pass


def check_json_dir(json_dir):
	"""TODO: Check '--json-dir' option for correctness"""
	pass


def time_ns(s):
	"""Convert time in nanoseconds as string to a number"""
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
@click.option('--pattern', help="JSON file name pattern, using {n} placeholder",
              default="sample_{n}.json", show_default=True)
@click.option('--size', '--n_objects', 'size_arg',
              help="Number of objects in JSON file to use, or 'scan'",
              default="scan", show_default=True)
@click.option('--output-csv', # uses path_type=click.Path (and not click.File) to support '--append'
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help="Output file in CSV format",
              default="benchmark.csv", show_default=True)
@click.option('--append/--no-append', default=False,
              help="Append to output file (no header)")
## These arguments should follow arguments of the benchmark executable
@click.option('--ws', '--workspace-size',
              help='Workgroup size.',
              type=click.Choice(['32','16','8','4']), show_choices=True,
              default='32', show_default=True)
@click.option('--const-order',
              help='Assumption of keys in JSON in a constant order',
              type=click.Choice(['0', '1']), show_choices=True,
              default='0', show_default=True)
@click.option('-V', '--version', # NOTE: conflicts with same option for version of script
              help='Version of dynamic string parsing.',
              type=click.IntRange(1, 3), # inclusive
              default='1', show_default=True)
@click.option('-s', '--max-string-size', 'str_size',
              help='Bytes allocated per dynamic string.  Turns on dynamic strings.',
              type=click.IntRange(min=1))
def main(exec_path, json_dir, pattern, size_arg, output_csv, append,
         ws, const_order, version, str_size):
    ### run as script

	click.echo(f"Using '{click.format_filename(exec_path)}' executable")
	click.echo(f"('{exec_path.resolve()}')")
	click.echo(f"  --workspace-size={ws}")
	click.echo(f"  --const-order={const_order}")
	if str_size is not None:
		click.echo(f"  --max-string-size={str_size}")
		click.echo(f"  --version={version}")
	else:
		click.echo(f"  --version={version} (ignored)")

	check_exec(exec_path)
	click.echo(f"JSON files from '{click.format_filename(json_dir)}' directory")
	check_json_dir(json_dir)
	
	if size_arg.isdigit():
		sizes = [int(size_arg, base=10)]
	elif size_arg == 'scan':
		sizes = range(100000, 900000+1, 100000)
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

	with click.progressbar(sizes, label='number of objects') as sizes_list:
		for size in sizes_list:
			json_file = json_dir / pattern.format(n=size)
			exec_args = [
				exec_path, json_file, str(size),
				f"--workspace-size={ws}",
				f"--const-order={const_order}",
				f"--version={version}",
			]
			if str_size is not None:
				exec_args.append(f"--max-string-size={str_size}")
			process = subprocess.Popen(
				exec_args,
				stdout=subprocess.PIPE
			)
			lines = process.stdout.read().decode('utf-8').split('\n')
			result = {
				'json file': json_file.name,
				'number of objects': size,
				'max string size': str_size,
			}

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
	re_string_handling = re.compile('Using (.*) string copy')
	re_assumptions     = re.compile('Assumptions: (.*)')
	re_workgroup_size  = re.compile('Workgroup size: W([0-9]*)')
	re_initialization  = re.compile('\\+ Initialization:\\s*([0-9.]*) ns')
	re_memory          = re.compile('\\+ Memory allocation and copying:\\s*([0-9.]*) ns')
	re_newlines        = re.compile('\\+ Finding newlines offsets (indices):\\s*([0-9.]*) ns')
	re_parsing_total   = re.compile('\\+ Parsing total (sum of the following):\\s*([0-9.]*) ns')
	re_json_processing = re.compile('  - JSON processing:\\s*([0-9.]*) ns')
	re_post_processing = re.compile('  - Post kernel hooks:\\s*([0-9.]*) ns')
	re_copyig_output   = re.compile('\\+ Copying output\\s*([0-9.]*) ns')
	re_gpu_total       = re.compile('Total time measured by GPU:\\s*([0-9.]*) ns')
	re_cpu_total       = re.compile('Total time measured by CPU:\\s*([0-9.]*) ns')

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
			result['Parsing total [ns]'] = time_ns(match.group(1))

		match = re_json_processing.match(line)
		if match:
			result['JSON processing [ns]'] = time_ns(match.group(1))

		match = re_post_processing.match(line)
		if match:
			result['Post kernel hooks [ns]'] = time_ns(match.group(1))

		match = re_copyig_output.match(line)
		if match:
			result['Copying output [ns]'] = time_ns(match.group(1))

		match = re_gpu_total.match(line)
		if match:
			result['Total time measured by GPU [ns]'] = time_ns(match.group(1))

		match = re_cpu_total.match(line)
		if match:
			result['Total time measured by CPU [ns]'] = time_ns(match.group(1))

	return result


if __name__ == '__main__':
	main()