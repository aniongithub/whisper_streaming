FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as devcontainer

ARG DEBIAN_FRONTEND=noninteractive

# nvidia docker runtime env
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN apt-get update &&\
    apt-get install -y \
    build-essential gdb \
    wget \
    software-properties-common \
    git git-lfs python3-pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM devcontainer as runtime

ENTRYPOINT ["python3", "whisper_online_server.py", "--port", "${PORT}", "--warmup-file", "assets/jfk.flac", "--model", "large-v3-turbo", "--model_cache_dir", "./.cache", "--language", "en", "--vad", "--buffer_trimming_sec", "2"]