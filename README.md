# Meta JSON Parser

## Requirements

Meta-JSON-Parser is written in C++, and it uses [CMake][] build system
(version 3.18 or higher).  You need a C++ compiler that supports the C++17
standard.

Meta-JSON-Parser needs [CUDA Toolkit][CUDA], at least version 10.0,
to be installed for it to compile and to run.  The project uses the
following _header-only_ libraries from the CUDA SDK:
- [Thrust][]: Parallel algorithms library (in the style of STL)
- [CUB][]: Cooperative primitives for CUDA C++

Meta-JSON-Parser also requires the following libraries to be either
installed locally or system-wide, or fetched as submodules into
the `third_party/` subdirectory:

- [Boost.Mp11][mp11]: A C++11 metaprogramming library
- [GoogleTest][GTest]: Google's C++ testing and mocking framework<br />
  (only for `meta-json-parser-test` binary)
- [CLI11][]: Command line parser for C++11<br />
  (only for `meta-json-parser-benchmark` binary)

[CMake]: https://cmake.org/
[Thrust]: https://thrust.github.io/
[CUB]: https://nvlabs.github.io/cub/
[CUDA]: https://docs.nvidia.com/cuda/ "CUDA Toolkit Documentation"
[mp11]: https://www.boost.org/doc/libs/1_66_0/libs/mp11/doc/html/mp11.html
[GTest]: https://google.github.io/googletest/
[CLI11]: https://github.com/CLIUtils/CLI11

## RAPIDS (cuDF) comparison and integration

The [RAPIDS][] suite of libraries and APIs gives the ability to execute
end-to-end data science and analytics pipelines entirely on NVIDIA GPUs.
RAPIDS include the [cuDF][], a [pandas][]-like DataFrame manipulation
library for Python, that Meta-JSON-Parser intends to integrate with.

cuDF in turn uses the [libcudf][], a C++ GPU DataFrame library for
_loading_, joining, aggregating, filtering, and otherwise manipulating
data.  Meta-JSON-Parser can **optionally** make use of libcudf, either
to benchmark JSON parsing time against, or to integrate with the
RAPIDS ecosystem.

To configure the build system to compile `meta-json-parser-benchmark`
with the libcudf support, use:
```.sh
cmake -DUSE_LIBCUDF=1 ..
```

[RAPIDS]: https://rapids.ai/
[cuDF]: https://github.com/rapidsai/cudf
[pandas]: https://pandas.pydata.org/
[libcudf]: https://docs.rapids.ai/api/libcudf/stable/
