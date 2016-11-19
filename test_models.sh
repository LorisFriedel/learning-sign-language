#!/bin/sh

BIN_PATH=./build/bin
GEN_MODEL_DIR=./generated_models

PREFIXE=$1 # prefixe placé avant le pattern dans le nom du fichier .xml généré
SUFFIXE=$2 # suffixe placé après le pattern dans le nom du fichier .xml généré
TEST_DIR=$3 # dossier contenant les fichiers de test

# $1 sera le pattern format 8_16_32
function test_model {
	./run_learning.sh --test-only -t $TEST_DIR -m $GEN_MODEL_DIR/model_${PREFIXE}_$1_${SUFFIXE}.xml
}

test_model "4_4_4"
test_model "4_4_4_4"
test_model "8_8"
test_model "8_8_8"
test_model "16_16"
test_model "8_16_32"
test_model "32_16_8"
test_model "32"
test_model "32_32"
test_model "32_32_32"
test_model "64_64"
