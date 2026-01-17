# 変更: CUDA対応のベースイメージを利用 (Python込み)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 元の構成を維持
RUN apt-get update && apt-get install -y --no-install-recommends sudo

ARG CONTAINER_WORKDIR
ENV CONTAINER_WORKDIR=$CONTAINER_WORKDIR
WORKDIR $CONTAINER_WORKDIR

ARG CONTAINER_USER
ENV CONTAINER_USER=$CONTAINER_USER
RUN useradd -ms /bin/bash $CONTAINER_USER
RUN echo $CONTAINER_USER:password | chpasswd
RUN echo "$CONTAINER_USER   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN usermod -aG audio $CONTAINER_USER || true

# パスを修正 (親ディレクトリの entrypoint.sh を参照するため)
COPY --chown=$CONTAINER_USER --chmod=755 .devcontainer/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# パッケージインストール (aptキャッシュ削除を修正)
RUN apt-get update && apt-get install -y --no-install-recommends \
    x11-apps \
    fonts-ipafont \
    python3-tk \
    portaudio19-dev \
    alsa-utils \
    libasound2-plugins \
    libpulse0 \
    libsndfile1 \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# パスを修正 (docker-composeのcontextに基づく)
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN rm /requirements.txt