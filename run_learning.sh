#!/bin/sh

BIN_PATH=./build/bin

if [ ! -f $BIN_PATH/learning.exe ]; then
    ./build.sh
fi

$BIN_PATH/learning.exe "$@"
