#!/bin/sh

DATA_DIR=../Documents/ml
LOG_DIR=./log

function generate_all {
	./generate_models.sh loris $1 $DATA_DIR/loris_$1 > $LOG_DIR/gen_model_$1_i_loris.log
	./generate_models.sh thomas $1 $DATA_DIR/thomas_$1 > $LOG_DIR/gen_model_$1_i_thomas.log
	./generate_models.sh others $1 $DATA_DIR/others_but_thomas_$1 > $LOG_DIR/gen_model_$1_i_others.log
	./generate_models.sh others_but_thomas $1 $DATA_DIR/others_but_thomas_$1 > $LOG_DIR/gen_model_$1_i_others_but_thomas.log
}

# $1 = suffixe of models
generate_all $1
