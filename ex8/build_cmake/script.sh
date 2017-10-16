#!/bin/bash

echo cleaning
make clean

echo making again
make

sigma=0.5;
alpha=0.01;
beta=0.001;
which_image=gaudi.png

echo launching the code again
./main -i ~/cuda_ss17/images/$which_image -sigma $sigma -alpha $alpha -beta $beta


