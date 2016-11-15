//
// Created by loris on 9/18/16.
//

#pragma once

#include <string>

namespace Default {
    const std::string CASCADE_PATH = "./data/haarcascades/haarcascade_frontalface_alt.xml";
    const std::string INPUT = "0";

    const std::string MODEL_NAME = "model.xml";
    const std::string GENERATED_MODEL_DIR = "generated_models/";
    const std::string MODEL_PATH = GENERATED_MODEL_DIR + MODEL_NAME;

    const std::string LETTERS_DATA_PATH = "letters_data/";
    const std::string LETTERS_IMAGES_PATH = "letters_images/";

    const std::string KEY_LETTER = "letter";
    const std::string KEY_MAT = "mat";

    const int NB_OF_LAYER = 2;
    const int NB_OF_NEURON = 128;

    const int HOG_IMG_SIZE = 256;
    const int HOG_BLOCK_SIZE = 32;
    const int HOG_BLOCK_STRIDE_SIZE = 16;
    const int HOG_CELL_SIZE = 8;
}
