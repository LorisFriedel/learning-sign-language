#!/bin/sh

BIN_PATH=./build/bin

if [ ! -f $BIN_PATH/img_convert.exe ]; then
    ./build.sh
fi

$BIN_PATH/img_convert.exe "$@"
