# benchmark/examples

This directory includes example JSON files for parsing with
meta-json-parser-benchmark, together with fragments of code required
to parse them.

For example, to check that floating point numbers are parsed correctly,
one can do the following:

1. Copy the `float.data_def.cuh` file to `..` as `data_def.cuh`:
   `cp float.data_def.cuh ../data_def.cuh`.  Before this if needed
   back up existing `data_def.cuh`.
2. Recompile parser with `make meta-json-parser-benchmark` (this needs
   to be done in the `../../build/` directory).
3. Run parser from `../../build/` with, for example, the following:
   ```.sh
   ./meta-json-parser-benchmark [OPTIONS] \
       ../benchmark/examples/float.json \
	   $(wc -l <../benchmark/examples/float.json)
   ```
