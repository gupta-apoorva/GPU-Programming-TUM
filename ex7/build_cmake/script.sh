#!/bin/bash

echo cleaning
#make clean

echo making again
#make

sigma=0.3;
which_image=peacockls.png

echo launching the code again
./main -i ~/cuda_ss17/images/$which_image -sigma $sigma


