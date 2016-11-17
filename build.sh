#!/bin/sh

./clean.sh

start=`date +%s`
mkdir -p build
cd build

cmake ..
make -j
end=`date +%s`

runtime=$((end-start))
echo "Build done. ("${runtime}" s)"
