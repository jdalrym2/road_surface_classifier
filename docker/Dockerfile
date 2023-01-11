FROM nvidia/cuda:11.7.0-base-ubuntu22.04 as final
SHELL ["/bin/bash", "-c"]

# Install necessary software for Python
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    lcov \
    pkg-config \
    libreadline6-dev \
    libncurses5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    liblzma-dev \
    libc6-dev \
    libbz2-dev \
    libffi-dev \
    lzma \
    lzma-dev \
    uuid-dev \
    zlib1g-dev

# Compile Python
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tgz && \
    tar xf Python-3.10.6.tgz && \
    cd Python-3.10.6 && \
    ./configure && make && make install && \
    python3 -m venv --copies /venv && \
    cd /tmp && rm -rf Python-3.10.6.tgz Python-3.10.6

# Install necessary software for Python venv + GDAL
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y \
    libopenjp2-7-dev \
    swig \
    cmake \
    libjpeg-dev \
    libkml-dev \
    libgeos-dev \
    libhdf5-dev \
    libzstd-dev \
    libpq-dev \
    libnetcdf-dev \
    librttopo-dev \
    libcrypto++-dev \
    libjson-c-dev \
    libproj-dev \
    libxml2-dev \
    libgeotiff-dev \
    libexpat1-dev \
    libpng-dev \
    libhdf4-dev

# Python requirements
WORKDIR /tmp
ADD requirements.txt /tmp
RUN source /venv/bin/activate && \
    python3 -m pip install --no-cache-dir wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Compile & Install GDAL
WORKDIR /tmp
ADD lerc-master.tar.gz /tmp
RUN cd /tmp/lerc-master && \
    cmake . && make && make install && \
    cd /tmp && rm -rf lerc-master


ADD tiff-4.5.0.tar.gz /tmp/
RUN cd /tmp/tiff-4.5.0 && \
    cmake . && make && make install && \
    cd /tmp && rm -rf tiff-4.5.0

ADD gdal-3.5.0.tar.gz /tmp
RUN cd /tmp/gdal-3.5.0 && \
    source /venv/bin/activate && \
    cmake -S . -B build \
    -DCMAKE_PREFIX_PATH="/venv" \
    -DCMAKE_INSTALL_PREFIX="/venv" \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j 8 --target install && \
    rm -r /tmp/gdal-3.5.0*

# User info
ARG USERNAME=dev
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Software for dev
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    vim \
    git \
    sudo && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash -m $USERNAME && \
    echo dev ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Environment variables
ENV GDAL_DATA=/venv/share/gdal/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/venv/lib