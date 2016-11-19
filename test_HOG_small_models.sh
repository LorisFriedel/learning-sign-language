#!/bin/sh

./test_models_on_all_set.sh loris HOG_small
./test_models_on_all_set.sh thomas HOG_small
./test_models_on_all_set.sh others HOG_small
./test_models_on_all_set.sh others_but_thomas HOG_small
