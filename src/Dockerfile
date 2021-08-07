ARG BASE_IMAGE_ARG

FROM ${BASE_IMAGE_ARG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y git-all libssl-dev python3-dev python3-pip python3-setuptools cmake build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip setuptools wheel opencv-python jupyter jupyterlab && \
    python3 -m pip install --upgrade mxnet-cu112 autogluon gluonts

WORKDIR /autogluon 

ARG PYTHON_FILE