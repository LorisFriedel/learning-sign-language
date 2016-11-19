#!/bin/sh

BIN_PATH=./build/bin

PREFIXE=$1 # prefixe placé avant le pattern dans le nom du fichier .xml généré
INPUT_DIR=$2 # dossier contenant les fichiers d'entrainement

# $1 sera le pattern format 8_16_32
function generate_model {
	PATTERN=$(echo $1 | sed 's/_/ /g')
	./run_learning.sh --no-test -i $INPUT_DIR -p "${PATTERN}" -o ./generated_models/model_${PREFIXE}_$1.xml
}

generate_model "4_4_4"
generate_model "4_4_4_4"
generate_model "8_8"
generate_model "8_8_8"
generate_model "16_16"
generate_model "8_16_32"
generate_model "32_16_8"
generate_model "32"
generate_model "32_32"
generate_model "32_32_32"
generate_model "64_64"

