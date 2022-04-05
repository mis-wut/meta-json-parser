### Base image
FROM rapidsai/rapidsai-core-dev:21.10-cuda11.4-devel-ubuntu18.04-py3.8 AS base
#22.02-cuda11.5-devel-ubuntu20.04-py3.8
#21.12-cuda11.5-devel-ubuntu20.04-py3.8
RUN mv /opt/conda/envs/rapids/include/boost/mp11 /opt/conda/envs/rapids/include/boost/mp11_do_not_use

### CLion remote builder
FROM base AS clion-remote-builder
RUN apt-get update && apt-get install -y ssh gdb
RUN (echo $'#!/bin/bash \n\
    export PATH="/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \n\
    export NVARCH="x86_64" \n\
    export NVIDIA_REQUIRE_CUDA="cuda>=11.5 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450" \n\
    export NV_CUDA_CUDART_VERSION="11.5.50-1" \n\
    export NV_CUDA_COMPAT_PACKAGE="cuda-compat-11-5" \n\
    export CUDA_VERSION="11.5.0" \n\
    export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/lib:/opt/conda/envs/rapids/lib" \n\
    export NVIDIA_VISIBLE_DEVICES="all" \n\
    export NVIDIA_DRIVER_CAPABILITIES="compute,utility" \n\
    export NV_CUDA_LIB_VERSION="11.5.0-1" \n\
    export NV_NVTX_VERSION="11.5.50-1" \n\
    export NV_LIBNPP_VERSION="11.5.1.53-1" \n\
    export NV_LIBNPP_PACKAGE="libnpp-11-5=11.5.1.53-1" \n\
    export NV_LIBCUSPARSE_VERSION="11.7.0.31-1" \n\
    export NV_LIBCUBLAS_PACKAGE_NAME="libcublas-11-5" \n\
    export NV_LIBCUBLAS_VERSION="11.7.3.1-1" \n\
    export NV_LIBCUBLAS_PACKAGE="libcublas-11-5=11.7.3.1-1" \n\
    export NV_LIBNCCL_PACKAGE_NAME="libnccl2" \n\
    export NV_LIBNCCL_PACKAGE_VERSION="2.11.4-1" \n\
    export NCCL_VERSION="2.11.4-1" \n\
    export NV_LIBNCCL_PACKAGE="libnccl2=2.11.4-1+cuda11.5" \n\
    export NV_CUDA_CUDART_DEV_VERSION="11.5.50-1" \n\
    export NV_NVML_DEV_VERSION="11.5.50-1" \n\
    export NV_LIBCUSPARSE_DEV_VERSION="11.7.0.31-1" \n\
    export NV_LIBNPP_DEV_VERSION="11.5.1.53-1" \n\
    export NV_LIBNPP_DEV_PACKAGE="libnpp-dev-11-5=11.5.1.53-1" \n\
    export NV_LIBCUBLAS_DEV_VERSION="11.7.3.1-1" \n\
    export NV_LIBCUBLAS_DEV_PACKAGE_NAME="libcublas-dev-11-5" \n\
    export NV_LIBCUBLAS_DEV_PACKAGE="libcublas-dev-11-5=11.7.3.1-1" \n\
    export NV_LIBNCCL_DEV_PACKAGE_NAME="libnccl-dev" \n\
    export NV_LIBNCCL_DEV_PACKAGE_VERSION="2.11.4-1" \n\
    export NV_LIBNCCL_DEV_PACKAGE="libnccl-dev=2.11.4-1+cuda11.5" \n\
    export LIBRARY_PATH="/usr/local/cuda/lib64/stubs" \n\
    export CONDA_DIR="/opt/conda" \n\
    export LANG="en_US.UTF-8" \n\
    export LC_ALL="en_US.UTF-8" \n\
    export LANGUAGE="en_US:en" \n\
    export DEBIAN_FRONTEND="noninteractive" \n\
    export CC="/usr/bin/gcc" \n\
    export CXX="/usr/bin/g++" \n\
    export CUDAHOSTCXX="/usr/bin/g++" \n\
    export CUDA_HOME="/usr/local/cuda" \n\
    export CONDARC="/opt/conda/.condarc" \n\
    export RAPIDS_DIR="/rapids" \n\
    export DASK_LABEXTENSION__FACTORY__MODULE="dask_cuda" \n\
    export DASK_LABEXTENSION__FACTORY__CLASS="LocalCUDACluster" \n\
    export NCCL_ROOT="/opt/conda/envs/rapids" \n\
    export PARALLEL_LEVEL="16" \n\
    export CUDAToolkit_ROOT="/usr/local/cuda" \n\
    export CUDACXX="/usr/local/cuda/bin/nvcc" \n\
    '; cat /root/.bashrc ) > /root/tmp_bashrc && \
    mv /root/tmp_bashrc /root/.bashrc
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd
RUN yes password | passwd root
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

### Build meta parser
FROM base AS parser-build

RUN mkdir -p /opt/meta-json-parser
WORKDIR /opt/meta-json-parser

COPY third_parties third_parties/
COPY benchmark benchmark/
COPY test test/
COPY meta_cudf meta_cudf/
COPY include include/
COPY CMakeLists.txt CMakeLists.txt

RUN mkdir build
WORKDIR build

RUN /opt/conda/envs/rapids/bin/cmake -DUSE_LIBCUDF=1 -DLOCAL_LIB=1 ..
RUN make -j

### Build python binding
FROM parser-build AS python-binding

#COPY libmeta-cudf-parser-1.a libmeta-cudf-parser-1.a # for debug
ENV LD_LIBRARY_PATH="/opt/meta-json-parser/build:${LD_LIBRARY_PATH}"
COPY python_binding python_binding/
COPY meta_cudf/parser.cuh python_binding/
WORKDIR python_binding
RUN make -j
WORKDIR ..

