FROM python:3.10.9 as python
RUN python -m venv --copies /venv

FROM nvidia/cuda:11.7.0-base-ubuntu20.04 as final
COPY --from=python /venv /venv
COPY --from=python /usr/local/ /usr/local/
SHELL ["/bin/bash", "-c"]

# Install necessary software
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    build-essential \
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
    libtiff-dev \
    libproj-dev \
    libxml2-dev \
    libgeotiff-dev \
    libexpat1-dev \
    libpng-dev \
    libsqlite3-dev \
    libhdf4-dev \
    libtk

# Python requirements
WORKDIR /tmp
ADD requirements.txt /tmp
RUN source /venv/bin/activate && \
    python3 -m pip install --no-cache-dir wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Compile & Install GDAL
WORKDIR /tmp
ADD gdal-3.5.0.tar.gz /tmp
RUN source /venv/bin/activate && \
    cd /tmp/gdal-3.5.0 && \
    cmake -S . -B build \
    -DCMAKE_PREFIX_PATH:FILEPATH="/venv" \
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
    apt-get install -y \
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
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/venv/lib/
ENV PATH=/venv/bin:${PATH}
