#!/bin/bash

echo cleaning
make clean

echo making again
make

which_image=peacock.png


echo launching the code again
./main -i ~/cuda_ss17/images/$which_image

 


