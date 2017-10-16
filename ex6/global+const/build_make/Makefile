main: ../main.cu ../helper.cu ../helper.h Makefile
	nvcc -o main ../main.cu ../helper.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

