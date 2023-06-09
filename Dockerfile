FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.6.0
WORKDIR /modelscope/pytorch

RUN pip install --no-cache-dir \
    loguru \
    fastapi[all] && \
    rm -rf /root/.cache/pip/*

COPY . .
