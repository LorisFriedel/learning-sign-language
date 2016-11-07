#!/bin/sh

if [ ! -f ./bin/facedetect.exe ]; then
    ./build.sh
fi

./bin/facedetect.exe "$@"
