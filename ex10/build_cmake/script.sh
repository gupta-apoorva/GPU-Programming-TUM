#!/bin/bash

echo cleaning
make clean

echo making again
make

sigma=0.5;
rho=3;
iterations=3000;
tau=0.2;
alpha=0.9;
C=0.000005;
which_image=van-gogh.png

echo launching the code again
./main -i ~/cuda_ss17/images/$which_image -iter $iterations -C $C -alpha $alpha -sigma $sigma -rho $rho -tau $tau


