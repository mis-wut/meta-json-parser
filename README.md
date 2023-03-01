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

- [Boost.Mp11][mp11]: A C++11 metaprogramming library (minimum 1.73)
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

[RAPIDS](https://rapids.ai/start.html) is available as
[conda](https://conda.io) packages,
[docker images][dockerhub-rapidsai],
and [from source builds][cudf-source].

[dockerhub-rapidsai]: https://hub.docker.com/r/rapidsai/rapidsai
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

### Installing libcudf as Docker image (and using it)

The RAPIDS images are based on [nvidia/cuda][], and are intended to be
drop-in replacements for the corresponding CUDA images
in order to make it easy to add RAPIDS libraries
while maintaining support for existing CUDA applications.

RAPIDS images come in three types, distributed in two different repos:
- `base` - contains a RAPIDS environment ready for use.
- `runtime` - extends the base image by adding a notebook server
  (JupyterLab) and example notebooks.
- **`devel`** - contains the full RAPIDS source tree,
  pre-built with all artifacts in place, and the compiler toolchain,
  the debugging tools, the headers and the static libraries for RAPIDS development.

[nvidia/cuda]: https://hub.docker.com/r/nvidia/cuda

#### Prerequisites

- NVIDIA Pascal GPU architecture (compute capability 6.1) or better
- CUDA 10.1+ with a compatible NVIDIA driver
- Ubuntu 18.04/20.04 or CentOS 7/8
- [Docker](https://www.docker.com/) CE v18+
- [nvidia-container-toolkit][]

[nvidia-container-toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

#### Installing Docker Engine

Docker Engine is an open source containerization technology
for building and containerizing your applications.  To install
it on Linux, follow distribution-specific [documentation][docker-docs]

[docker-docs]: https://docs.docker.com/engine/install/#server

For example on [Debian](https://docs.docker.com/engine/install/debian/)
installing Docker CE takes the following steps:

1. Uninstall old versions (that were not working correctly):
   ```shell
   $ sudo apt-get remove docker docker-engine docker.io containerd runc
   ```

2. Update the `apt` package index and install packages
   to allow `apt` to use a repository over HTTPS:
   ```shell
   $ sudo apt-get update
   $ sudo apt-get install \
      ca-certificates \
      curl \
      gnupg \
      lsb-release
   ```

3. Add Docker’s official GPG key for signing Debian packages:
   ```shell
   $ curl -fsSL https://download.docker.com/linux/debian/gpg | 
      sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   ```

4. Set up the stable package repository for Docker
   ```shell
   $ echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | 
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```
   Note that for Debian unstable you might need to explicitly use the
   latest stable version instead (bullseye):
   ```shell
   $ echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bullseye stable" | 
     sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
   ```

5. Update the `apt` package index again,
   ```shell
   $ sudo apt-get update
   ```
   checking that proper repository is used:
   ```
   Get:6 https://download.docker.com/linux/debian bullseye InRelease
   Get:7 https://download.docker.com/linux/debian bullseye/stable amd64
   Reading package lists... Done
   ```

6. Install the _latest version_ of Docker Engine and containerd:
   ```shell
   $ sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```
   You can check that the correct version of Docker was installed with
   ```shell
   $ apt-cache madison docker-ce
   ```
   which should return results from https://download.docker.com/linux/debian

You can check that Docker Engine is installed and runs correctly with
```
$ sudo docker run hello-world
```
You can also use
```
$ sudo docker ps -a
```
to see what Docker images are running and what images are installed.

As an optional post-installation step on Linux, you can
[configure Docker for use as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

#### Installing NVIDIA Docker Engine

The NVIDIA Container Toolkit allows users to build and run GPU accelerated containers.
The toolkit includes a container runtime [library](https://github.com/NVIDIA/libnvidia-container)
and utilities to automatically configure containers to leverage NVIDIA GPUs.

See [NVIDIA Container Toolkit Installation Guide][nvidia-container] or
[NVIDIA Docker Engine wrapper repository][nvidia-docker] for details.
  
[nvidia-container]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
[nvidia-docker]: https://nvidia.github.io/nvidia-docker/

1. Add NVIDIA’s official GPG key for signing packages:
   ```shell
   $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |
      sudo apt-key add -
   ```

2. Setup the `stable` repository
   ```shell
   $ distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   ```
   If this did not work, you may need to set up Linux distribution
   and its version manually, for example:
   ```shell
   $ curl -s -L https://nvidia.github.io/nvidia-docker/debian11/nvidia-docker.list |
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   ```
   (which, as you can see, returns incorrect URLs?).

3. Update the `apt` package index,
   ```shell
   $ sudo apt-get update
   ```
   checking that NVIDIA container repository is listed among repositories
   (https://nvidia.github.io/).

4. Install the `nvidia-docker2` package (and dependencies):
   ```shell
   $ sudo apt-get install nvidia-docker2
   ```

5. Restart the Docker daemon to complete the installation,
   and check that it works correctly:
   ```shell
   $ sudo systemctl restart docker
   $ sudo systemctl status docker
   ```

At this point, a working setup can be tested by running a base CUDA container:
```shell
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### Workaround for cgroups bug

If running a CUDA container fails with the following error message
```
docker: Error response from daemon: OCI runtime create failed: container_linux.go:380: starting container process caused:
process_linux.go:545: container init caused:
Running hook #0:: error running hook: exit status 1, stdout: , stderr: nvidia-container-cli:
container error: cgroup subsystem devices not found: unknown.
```
this might mean that the hierarchical v2 cgroups are used.

There are [two possible solutions][issues/1447]:
- turn off hierarchical cgroups by using 
  `systemd.unified_cgroup_hierarchy=false` kernel command line parameter, or
- turn off using cgroups by setting `no-cgroups = true` in
  `/etc/nvidia-container-runtime/config.toml`, and adding NVIDIA devices
  manually to the container
  ```shell
  $ sudo docker run --rm --gpus all \
     --device /dev/nvidia0 --device /dev/nvidia-modeset  \
     --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
     --device /dev/nvidiactl \
     nvidia/cuda:11.0-base nvidia-smi
  ```

[issues/1447]: https://github.com/NVIDIA/nvidia-docker/issues/1447

#### Start RAPIDS container and notebook server

Use the [RAPIDS Release Selector](https://rapids.ai/start.html#get-rapids),
choose "_Docker + Dev Env_", and appropriate switches, to find the correct
invocation:
```shell
$ docker pull rapidsai/rapidsai-core-dev:22.02-cuda11.5-devel-ubuntu20.04-py3.9
$ docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai-core-dev:22.02-cuda11.5-devel-ubuntu20.04-py3.9
```

**Note** that running `docker` might require using `sudo`, and that with
the cgroups2 workaround (cgroups disabled) one also needs to add appropriate
`--device` options, see above.

The following ports are used by the `runtime` and `core-dev` containers only
(not `base` containers):

- 8888 - exposes a [JupyterLab][] notebook server
- 8786 - exposes a [Dask] scheduler
- 8787 - exposes a [Dask diagnostic web server][dask-diag]

Read more at [RAPIDS at DockerHub][dockerhub-rapidsai].

[JupyterLab]: https://jupyterlab.readthedocs.io/en/stable/
[Dask]: https://docs.dask.org/en/latest/
[dask-diag]: https://docs.dask.org/en/latest/setup/cli.html#diagnostic-web-servers

#### Start RAPIDS container for meta-json-parser

To have access to the `meta-json-parser` project in the RAPIDS Docker container,
you can mount the directory on host with this project to specific location
within container (using bind mount).

Running the container can be done like this:
```shell
$ sudo docker run --gpus all --rm -it \
   --device /dev/nvidia0 --device /dev/nvidiactl \
   --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
   -p 8888:8888 -p 8787:8787 -p 8786:8786 \
   -v ${HOME}/GPU-IDUB/meta-json-parser:/meta-json-parser \
   -v ${HOME}/GPU-IDUB/data:/data \
   rapidsai/rapidsai-core-dev:21.12-cuda11.5-devel-ubuntu20.04-py3.8
```
Then you just need to change the directory in the container:
```
(rapids) root@8c8501d8b358:/rapids/notebooks# cd /meta-json-parser/build/
```
Before running `cmake` you might need to remove its cache; simplest
solution is to clean the `build` directory; it might be enough to
just remove `CMakeCache.txt`.

To configure the build system to compile `meta-json-parser-benchmark`
with the libcudf support, use:
```shell
(rapids) root@8c8501d8b358:/meta-json-parser/build# cmake -DUSE_LIBCUDF=1 ..
```

Note that Boost installed in a RAPIDS Docker contained might be too old:
```
-- Could NOT find Boost: Found unsuitable version "1.72.0", but required is at least "1.73" (found /opt/conda/envs/rapids/include, )
```

**Workaround** for `cmake`/`make`/`g++` using system-installed Boost:
```shell
(rapids) root@ccefed838be5:/meta-json-parser/build# cd /opt/conda/envs/rapids/include/boost
(rapids) root@ccefed838be5:/opt/conda/envs/rapids/include/boost# mv mp11 mp11_do_not_use
(rapids) root@ccefed838be5:/opt/conda/envs/rapids/include/boost# cd -
```
or simply
```
( cd /opt/conda/envs/rapids/include/boost ;  mv mp11 mp11_do_not_use )
```

You need to also use local libraries from `third_parties/` with
```shell
cmake -DUSE_LIBCUDF=1 -DLOCAL_LIB=1 ..
```

#### Provide CUDA runtime to `docker build`

To use CUDA runtime in `docker build` you need to install
[nvidia-container-runtime](https://github.com/nvidia/nvidia-container-runtime#installation).
[_source_](https://github.com/nvidia/nvidia-container-runtime#installation)

```shell
sudo apt-get install nvidia-container-runtime
```

CUDA runtime must be set as a default runtime in `/etc/docker/daemon.json`.
_If you installed nvidia-docker2, then nvidia runtime might be already configured._
_In that case provide just a default-runtime option._


```json
{
	"runtimes": {
		"nvidia": {
			"path": "/usr/bin/nvidia-container-runtime",
			"runtimeArgs": []
		}
	},
	"default-runtime": "nvidia"
}
```

It's [the only way](https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime)
to enable CUDA runtime in `docker build`.


## Citations

If you wish to cite this software in an academic publication,
please use the following reference:

Formatted:

- K. Kaczmarski, J. Narębski, S. Piotrowski and P. Przymus,
  "Fast JSON parser using metaprogramming on GPU,"
  2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA),
  Shenzhen, China, 2022, pp. 1-10,
  doi: 10.1109/DSAA54385.2022.10032381.

BibTeX:

```bibtex
@inproceedings{10032381,
  author={Kaczmarski, Krzysztof and Narębski, Jakub and Piotrowski, Stanisław and Przymus, Piotr},
  booktitle={2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA)},
  title={Fast JSON parser using metaprogramming on GPU},
  year={2022},
  pages={1-10},
  doi={10.1109/DSAA54385.2022.10032381}}
```
