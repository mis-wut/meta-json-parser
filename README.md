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

## Installing and using libcudf library

[RAPIDS](https://rapids.ai/start.html) is available as [conda](https://conda.io)
packages, docker images, and [from source builds][cudf-source].

[cudf-source]: https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#script-to-build-cudf-from-source

### Installing libcudf using conda (and using it)

To install Miniconda environment together with the `conda` tool on Linux, one
can install it from RPM and Debian (dpkg) repositories for Miniconda, as
described in the [RPM and Debian Repositories for Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/rpm-debian.html)
section of the [Conda User's Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html).

#### Install `conda` (if not present)

1. Download the public GPG key for conda repositories and add it to the keyring

   ```
   $ curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor >~/conda.gpg
   $ sudo install -o root -g root -m 644 ~/conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
   ```

2. Check whether fingerprint is correct (will output an error message otherwise)

   ```
   $ gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
   ```

3. Add conda repo to list of sources for apt

   ```
   $ echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" | sudo tee -a /etc/apt/sources.list.d/conda.list
   ```

4. Conda is ready to install

   ```
   $ sudo apt update
   $ sudo apt install conda
   ```

5. To use `conda` you need to configure some environment variables

   ```
   $ source /opt/conda/etc/profile.d/conda.sh
   ```

6. You can see if the installation was successful by typing

   ```
   $ conda -V
   conda 4.9.2
   ```

#### Install cuDF using `conda`

1. The command to install RAPIDS libraries, cuDF in particular, can be found via
   [RAPIDS Release Selector](https://rapids.ai/start.html#get-rapids).

   The version 0.19 of the cuDF library (that was used in previous comparison)
   can be installed with:
   ```
   $ conda create -n rapids-0.19 -c rapidsai -c nvidia -c conda-forge \
                     cudf=0.19 python=3.8 cudatoolkit=11.2
   ```
   
   The current stable release of cuDF (as of January 2022), assuming Python 3.8
   (there is Python 3.9.9 on 'choinka'), and CUDA 11.4 (version installed on 'choinka')
   can be installed with:
   ```
   $ conda create -n rapids-21.12 -c rapidsai -c nvidia -c conda-forge \
                     cudf=21.12 python=3.8 cudatoolkit=11.4
   ```

   **Note** that this installs cuDF _locally_, for the currently logged in user.

> **Note**: to remove the created conda environment, use
> ```
> $ conda env remove -n rapids-0.19
> ```

#### Compiling and running `meta-json-parser-benchmark` using conda-installed cuDF

1. To use conda-installed cuDF (and libcudf), and to compile using this version
   of the library and its header files, one needs to activate 'rapids-21.12'
   environment with `conda` (as told by `conda create ...` command that was
   invoked in the previous step):

   ```
   $ conda activate rapids-21.12
   ```

   This commands, among other things, set-ups environment variables (including
   `CONDA_PREFIX`) and modifies shell prompt to include information about
   current conda environment. You can then recompile the project with `make
   clean && make`.

   To turn off using cuDF, run `conda deactivate`.

2. To run `meta-json-parser-benchmark` with conda-installed libcudf, one needs to have
   'rapids-21.22' environment active (see the previous step), at least to use the
   command as given here.

   For `meta-json-parser-benchmark` to use conda-installed libcudf, and to prefer it over
   system-installed version (if any), one needs to set `LD_LIBRARY_PATH`
   environment variable correctly. It needs to include the path to
   conda-installed libcudf library, but also paths to libraries used by
   `./meta-json-parser-benchmark` (you can find them with [`ldd`][ldd] command), for example:

   **TODO**: check for correctness.
   ```
   LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/lib/x86_64-linux-gnu:/lib64" \
      ./meta-json-parser-benchmark ../../data/json/generated/sample_400000.json 400000 \
      --max-string-size=32 --const-order  -o sample_b.csv
   ```

[ldd]: https://man7.org/linux/man-pages/man1/ldd.1.html "ldd(1) - Linux manual page"