# 0.2.0-rapids0.19-cuda10.2-framework-ubuntu18.04-py3.7
#  ^          ^        ^    ^          ^          ^
#  |          |        |    type       |          python version
#  |          |        |               |
#  |          |        cuda version    |
#  |          |                        |
#  |          RAPIDS version           linux version
#  |
#  AutoGluon version
ARG AUTOGLUON_VERSION=0.2.0
ARG RAPIDS_VERSION=0.19
ARG CUDA_VERSION=10.2
ARG BUILD_TYPE=jupyter
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7

FROM autogluon/autogluon:${AUTOGLUON_VERSION}-rapids${RAPIDS_VERSION}-cuda${CUDA_VERSION}-jupyter-${BUILD_TYPE}-py${PYTHON_VERSION}