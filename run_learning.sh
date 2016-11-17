#!/bin/sh

if [ ! -f ./build/bin/learning.exe ]; then
    ./build.sh
fi

./build/bin/learning.exe "$@"
