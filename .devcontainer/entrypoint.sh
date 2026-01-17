#!/bin/bash

export HOME=/home/$CONTAINER_USER

# --- 既存のワークスペースUID/GID調整ロジック ---
uid=$(stat -c "%u" $CONTAINER_WORKDIR)
gid=$(stat -c "%g" $CONTAINER_WORKDIR)

if [ "$uid" -ne 0 ]; then
    if [ "$(id -u $CONTAINER_USER)" -ne $uid ]; then
        EXISTING_USER=$(id -nu $uid)
        if [ $EXISTING_USER ] ; then
            userdel -r $EXISTING_USER
        fi
        usermod -u $uid $CONTAINER_USER
    fi
    if [ "$(id -g $CONTAINER_USER)" -ne $gid ]; then
        EXISTING_GROUP=$(id -ng $gid)
        if [ $EXISTING_GROUP ] ; then
            groupdel $EXISTING_GROUP
        fi
        getent group $gid >/dev/null 2>&1 || groupmod -g $gid $CONTAINER_USER
        chgrp -R $gid $HOME
    fi
fi

# --- 【追加】GPUデバイス権限の動的割り当て ---
# ホスト側の /dev/kfd と /dev/dri/renderD128 のグループIDを取得し、
# コンテナ内ユーザーをそのグループに追加する。

# 対象のデバイスリスト
GPU_DEVICES=("/dev/kfd" "/dev/dri/renderD128" "/dev/dri/card0" "/dev/dri/card1")

for dev in "${GPU_DEVICES[@]}"; do
    if [ -e "$dev" ]; then
        # デバイスの所有グループID(GID)を取得
        dev_gid=$(stat -c "%g" "$dev")
        
        # すでにそのGIDを持つグループが存在するか確認
        if getent group "$dev_gid" >/dev/null; then
            # 存在する場合、そのグループ名を取得
            target_group=$(getent group "$dev_gid" | cut -d: -f1)
        else
            # 存在しない場合、新しくグループを作成 (名前は適当に gpu_group_GID)
            target_group="gpu_group_$dev_gid"
            groupadd -g "$dev_gid" "$target_group"
        fi

        # ユーザーをそのグループに追加 (すでに所属していてもエラーにならない)
        usermod -aG "$target_group" "$CONTAINER_USER"
    fi
done

# --- コマンド実行 ---
exec setpriv --reuid=$CONTAINER_USER --regid=$CONTAINER_USER --init-groups "$@"