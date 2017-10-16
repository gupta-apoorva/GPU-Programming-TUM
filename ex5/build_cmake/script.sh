#!/bin/bash

echo cleaning
make clean

echo making again
make

sigma=6.32;
which_image=flowers.png


echo launching the code again
./main -i ~/cuda_ss17/images/$which_image -sigma $sigma 


