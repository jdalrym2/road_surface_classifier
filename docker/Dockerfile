FROM condaforge/mambaforge:4.12.0-0 as conda
SHELL ["/bin/bash", "-c"]

# Create conda environment
COPY environment.yml .
RUN mamba env create -n tmp_env --file environment.yml && \
    mamba install conda-pack

# Install build tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    build-essential

# Install GDAL into conda environment
WORKDIR /tmp
ADD gdal-3.5.0.tar.gz /tmp
RUN source /opt/conda/bin/activate tmp_env && \
    cd /tmp/gdal-3.5.0 && \
    cmake -S . -B build \
    -DCMAKE_PREFIX_PATH:FILEPATH="/opt/conda/envs/tmp_env" \
    -DCMAKE_INSTALL_PREFIX="/opt/conda/envs/tmp_env" \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j 8 --target install && \
    rm -r /tmp/gdal-3.5.0*

# Pack conda environment for delivery
RUN source /opt/conda/bin/activate && \
    conda-pack -n tmp_env -o /tmp/env.tar && \
    mkdir /venv && \
    cd /venv && tar -xf /tmp/env.tar && cd - && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

FROM debian:bullseye as final
SHELL ["/bin/bash", "-c"]

ARG USERNAME=dev
ARG USER_UID=1001
ARG USER_GID=$USER_UID

COPY --from=conda --chmod=0755 /venv /venv/
ENV GDAL_DATA=/venv/share/gdal/
ENV LD_LIBRARY_PATH=/venv/lib/

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
