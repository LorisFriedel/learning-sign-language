#!/bin/sh

if [ ! -f ./bin/learning.exe ]; then
    ./build.sh
fi

./bin/learning.exe "$@"
