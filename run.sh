#!/bin/sh
mkdir -p out
python ./main.py > out/diffusion.dat
grep "norm" out/diffusion.dat | cut -d '=' -f 2 > out/norm.dat
