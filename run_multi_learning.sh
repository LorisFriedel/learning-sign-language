#!/bin/sh

BIN_PATH=./build/bin

if [ ! -f $BIN_PATH/multi_learning.exe ]; then
    ./build.sh
fi

$BIN_PATH/multi_learning.exe "$@"
