#
#
# Practical Course: GPU Programming in Computer Vision
#
#
# Technical University Munich, Computer Vision Group
# Winter Semester 2015/2016, March 15 - April 15
#
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
CUDA_ADD_EXECUTABLE(main main.cu helper.cu)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})

