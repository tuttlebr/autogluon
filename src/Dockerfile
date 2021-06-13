ARG BASE_IMAGE_ARG
FROM ${BASE_IMAGE_ARG}

ARG EXPERIMENT_TYPE
ARG MXNET
ARG NB_PREFIX=/

ENV HOME=/home/${EXPERIMENT_TYPE}
ENV DEBIAN_FRONTEND noninteractive

WORKDIR ${HOME}

COPY clean-layer.sh /usr/bin/clean-layer.sh
COPY bashrc /etc/bash.bashrc
COPY ${EXPERIMENT_TYPE}/requirements.txt .

RUN chmod a+rwx /usr/bin/clean-layer.sh \
    && chmod a+rwx /etc/bash.bashrc \
    && apt-get -yq update \
    && apt-get install -y --no-install-recommends \
    curl \
    g++ \
    gcc \
    build-essential \
    nano \
    && clean-layer.sh \
    && curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
    && echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get -yq update \
    && apt-get -yq install --no-install-recommends \
    nodejs \
    && clean-layer.sh \
    && python3 -m pip install -U pip \
    && python3 -m pip install -U ${MXNET} \
    && python3 -m pip install --no-cache-dir -r requirements.txt

CMD ["sh","-c", "jupyter lab --notebook-dir=${HOME} --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]