#!/bin/sh

if [ ! -f ./build/bin/camshift.exe ]; then
    ./build.sh
fi

./build/bin/camshift.exe "$@"
