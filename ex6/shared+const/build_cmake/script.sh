#!/bin/bash

echo cleaning
make clean

echo making again
make

sigma=1 #should be less than 6.66 to maintain the rmax under 20
which_image=florida.png

echo launching the code again
./main -i ~/cuda_ss17/images/$which_image -sigma $sigma


