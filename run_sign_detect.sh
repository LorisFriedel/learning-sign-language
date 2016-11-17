#!/bin/sh

BIN_PATH=./build/bin

if [ ! -f $BIN_PATH/sign_detect.exe ]; then
    ./build.sh
fi

$BIN_PATH/sign_detect.exe "$@"
