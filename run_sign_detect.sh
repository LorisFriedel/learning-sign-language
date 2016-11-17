#!/bin/sh

if [ ! -f ./build/bin/sign_detect.exe ]; then
    ./build.sh
fi

./build/bin/sign_detect.exe "$@"
