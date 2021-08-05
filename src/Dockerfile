ARG BASE_IMAGE_ARG

FROM ${BASE_IMAGE_ARG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-opencv \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install -U pip && \
    python3 -m pip install -U setuptools wheel opencv-python && \
    python3 -m pip install -U "mxnet_cu112<2.0.0" && \
    git clone https://github.com/awslabs/autogluon  && \
    cd autogluon && ./full_install.sh

WORKDIR /autogluon 

ARG PYTHON_FILE

ENTRYPOINT ["python3", "/autogluon/sample.py"]