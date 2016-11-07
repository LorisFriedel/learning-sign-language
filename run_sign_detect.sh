#!/bin/sh

if [ ! -f ./bin/sign_detect.exe ]; then
    ./build.sh
fi

./bin/sign_detect.exe "$@"
