#!/bin/sh


./generate_models.sh loris ../Documents/ml/loris_img_HOG > model_i_loris_gen.log
./generate_models.sh thomas ../Documents/ml/thomas_img_HOG > model_i_thomas_gen.log
./generate_models.sh others_but_thomas ../Documents/ml/others_but_thomas_img_HOG > model_i_others_but_thomas_gen.log

./test_models.sh loris ../Documents/ml/loris_img_HOG > test_model_loris_gen.log
./test_models.sh thomas ../Documents/ml/thomas_img_HOG > test_model_thomas_gen.log
./test_models.sh others_but_thomas ../Documents/ml/others_but_thomas_img_HOG > test_model_others_but_thomas_gen.log
