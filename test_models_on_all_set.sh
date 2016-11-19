#!/bin/sh

DATA_DIR=../Documents/ml
LOG_DIR=./log

function test_model {
	./test_models.sh $1 $2 $DATA_DIR/loris_$2 > $LOG_DIR/test_model_$2_$1_on_loris.log
	./test_models.sh $1 $2 $DATA_DIR/others_$2 > $LOG_DIR/test_model_$2_$1_on_others.log
	./test_models.sh $1 $2 $DATA_DIR/thomas_$2 > $LOG_DIR/test_model_$2_$1_on_thomas.log
	./test_models.sh $1 $2 $DATA_DIR/others_but_thomas_$2 > $LOG_DIR/test_model_$2_$1_on_others_but_thomas.log
}

# $1 = modele a tester, $2 suffixe des modeles
test_model $1 $2
