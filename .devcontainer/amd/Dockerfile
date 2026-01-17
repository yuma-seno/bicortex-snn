# 変更: ROCm対応のベースイメージを利用
FROM rocm/pytorch:latest

# 元の構成を維持
RUN apt-get update && apt-get install -y --no-install-recommends sudo

ARG CONTAINER_WORKDIR
ENV CONTAINER_WORKDIR=$CONTAINER_WORKDIR
WORKDIR $CONTAINER_WORKDIR

ARG CONTAINER_USER
ENV CONTAINER_USER=$CONTAINER_USER
# ROCmイメージによってはユーザIDの扱いが特殊な場合があるため強制的に作成
RUN useradd -ms /bin/bash $CONTAINER_USER || echo "User exists"
RUN echo $CONTAINER_USER:password | chpasswd
RUN echo "$CONTAINER_USER   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# videoグループへの追加がROCmでは重要
RUN usermod -aG audio,video,render $CONTAINER_USER || true

COPY --chown=$CONTAINER_USER --chmod=755 .devcontainer/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

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

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN rm /requirements.txt