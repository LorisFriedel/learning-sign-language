#!/bin/sh

./clean.sh

cmake ./ && make
echo "Build done."