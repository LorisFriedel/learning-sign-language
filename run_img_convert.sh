#!/bin/sh

if [ ! -f ./bin/img_convert.exe ]; then
    ./build.sh
fi

./bin/img_convert.exe "$@"
