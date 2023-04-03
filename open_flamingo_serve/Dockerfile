ARG TORCH_PACKAGE_VERSION=1.12.1
ARG CUDA_VERSION=11.3
ARG CUDNN_VERSION=8


FROM pytorch/pytorch:${TORCH_PACKAGE_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]