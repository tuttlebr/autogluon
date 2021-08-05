ARG BASE_IMAGE_ARG

FROM ${BASE_IMAGE_ARG}

ARG MXNET

ENV DEBIAN_FRONTEND noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /autogluon

COPY object-detection.py .

RUN source activate rapids && \
    python3 -m pip install -U pip && \
    python3 -m pip install -U setuptools wheel && \
    python3 -m pip install -U "${MXNET}" && \
    python3 -m pip install autogluon

CMD ["/bin/bash", "-c", "python3 /autogluon/object-detection.py"]