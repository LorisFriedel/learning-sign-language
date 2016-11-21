#!/bin/sh

DATA_DIR=../Documents/ml
LOG_DIR=./log

function generate_all {
	./generate_models.sh others $1 $DATA_DIR/others_$1 > $LOG_DIR/gen_model_$1_i_others.log
}

# $1 = suffixe of models
generate_all yml
generate_all HOG
generate_all HOG_small
