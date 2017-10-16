#!/bin/bash

echo cleaning
make clean

echo making again
make

iterations=100
tau=0.2
epsilon=0.1
g_type=0
which_image=flowers.png

echo launching the code again
./main -i ~/cuda_ss17/images/$which_image -iter $iterations -tau $tau -epsilon $epsilon -g_type $g_type 

# -gtype  0 ....  for g =1
# -gtype  1 ....  for max g_type
# -gtype  2 ....  for eps g_type
