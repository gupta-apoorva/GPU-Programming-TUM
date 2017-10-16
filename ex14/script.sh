#!/bin/bash

echo MAKING

nvcc cublas_v2.cu -lcublas -o cublas_v2
nvcc cublas_v1.cu -lcublas -o cublas_v1
nvcc custom_add.cu -o custom_add

echo
echo
echo

echo ++++++++++++++++++++++++++++++++++++++++++++++++++++

echo
echo

echo LAUNCHING CUSTOM ADD FUNCTION FOR ARRAY SIZE OF 10^6
./custom_add

echo 

echo LAUNCHING NEW CUBLAS ADD FUNCTION FOR ARRAY SIZE OF 10^6
./cublas_v2

echo

echo LAUNCHING OLD CUBLAS ADD FUNCTION FOR ARRAY SIZE OF 10^6
./cublas_v1

echo
echo

echo ++++++++++++++++++++++++++++++++++++++++++++++++++++

echo
echo
echo






