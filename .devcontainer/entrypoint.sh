#!/bin/bash

export HOME=/home/$CONTAINER_USER

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

exec setpriv --reuid=$CONTAINER_USER --regid=$CONTAINER_USER --init-groups "$@"