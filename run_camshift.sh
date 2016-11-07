#!/bin/sh

if [ ! -f ./bin/camshift.exe ]; then
    ./build.sh
fi

./bin/camshift.exe "$@"
