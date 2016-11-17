#!/bin/sh

if [ ! -f ./build/bin/img_convert.exe ]; then
    ./build.sh
fi

./build/bin/img_convert.exe "$@"
