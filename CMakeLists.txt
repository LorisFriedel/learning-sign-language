cmake_minimum_required(VERSION 3.6)
project(TP_opencv)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

set(SRC_DIR src)
set(INC_DIR inc)
set(LIB_DIR lib)

include_directories(${LIB_DIR})

set(EXEC_FACEDETECT facedetect.exe)
set(EXEC_CAMSHIFT camshift.exe)
set(EXEC_SIGN_DETECT sign_detect.exe)
set(EXEC_LEARNING learning.exe)

set(MAIN_FACEDETECT src/main_facedetect.cpp)
set(MAIN_CAMSHIFT src/main_camshift.cpp)
set(MAIN_SIGN_DETECT src/main_sign_detect.cpp)
set(MAIN_LEARNING src/main_learning.cpp)

set(SRC_FACEDETECT inc/constant.h src/ObjectDetector.cpp inc/ObjectDetector.hpp src/VideoStreamReader.cpp inc/VideoStreamReader.hpp inc/geo.h inc/log.h inc/code.h inc/colors.h inc/time.h src/ObjectDetectRunner.cpp inc/ObjectDetectRunner.hpp)
set(SRC_CAMSHIFT inc/constant.h src/ObjectDetector.cpp inc/ObjectDetector.hpp src/VideoStreamReader.cpp inc/VideoStreamReader.hpp inc/geo.h inc/log.h inc/code.h src/CamshiftTracker.cpp inc/CamshiftTracker.hpp src/KeyInputHandler.cpp inc/KeyInputHandler.hpp src/CamshiftRunner.cpp inc/CamshiftRunner.hpp inc/colors.h src/HandTracker.cpp inc/HandTracker.hpp inc/time.h)
set(SRC_SIGN_DETECT inc/constant.h src/ObjectDetector.cpp inc/ObjectDetector.hpp src/VideoStreamReader.cpp inc/VideoStreamReader.hpp inc/geo.h inc/log.h inc/code.h src/CamshiftTracker.cpp inc/CamshiftTracker.hpp src/KeyInputHandler.cpp inc/KeyInputHandler.hpp src/CamshiftRunner.cpp inc/CamshiftRunner.hpp inc/colors.h src/HandTracker.cpp inc/HandTracker.hpp inc/time.h src/MLPHand.cpp inc/MLPHand.hpp)
set(SRC_LEARNING inc/constant.h inc/log.h inc/code.h  src/MLPHand.cpp inc/MLPHand.hpp)

set(EXECUTABLE_OUTPUT_PATH ./bin/)
add_executable(${EXEC_FACEDETECT} ${MAIN_FACEDETECT} ${SRC_FACEDETECT})
add_executable(${EXEC_CAMSHIFT} ${MAIN_CAMSHIFT} ${SRC_CAMSHIFT})
add_executable(${EXEC_SIGN_DETECT} ${MAIN_SIGN_DETECT} ${SRC_SIGN_DETECT})
add_executable(${EXEC_LEARNING} ${MAIN_LEARNING} ${SRC_LEARNING})

target_link_libraries(${EXEC_FACEDETECT} ${OpenCV_LIBS})
target_link_libraries(${EXEC_CAMSHIFT} ${OpenCV_LIBS})
target_link_libraries(${EXEC_SIGN_DETECT} ${OpenCV_LIBS})
target_link_libraries(${EXEC_LEARNING} ${OpenCV_LIBS})
