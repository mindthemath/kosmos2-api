# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.2.2-base-ubuntu22.04
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python$PYTHON_VERSION get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python \
    && ln -sf /usr/local/bin/pip$PYTHON_VERSION /usr/local/bin/pip \
    && python --version \
    && pip --version \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


ENV HF_ENABLE_HF_TRANSFER=1
EXPOSE 8020
COPY server.py /app
CMD ["python", "/app/server.py"]
