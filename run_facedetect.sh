#!/bin/sh

BIN_PATH=./build/bin

if [ ! -f $BIN_PATH/facedetect.exe ]; then
    ./build.sh
fi

$BIN_PATH/facedetect.exe "$@"
