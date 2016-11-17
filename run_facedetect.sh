#!/bin/sh

if [ ! -f ./build/bin/facedetect.exe ]; then
    ./build.sh
fi

./build/bin/facedetect.exe "$@"
