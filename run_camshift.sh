#!/bin/sh

BIN_PATH=./build/bin

if [ ! -f $BIN_PATH/camshift.exe ]; then
    ./build.sh
fi

$BIN_PATH/camshift.exe "$@"
