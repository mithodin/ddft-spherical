#!/bin/bash
if [ ! -f "vanhove.h5" ]; then
  wget "https://files.treffenstaedt.de/vanhove.h5" -O vanhove.h5
fi
mkdir -p out
python3 ./main.py > out/diffusion-test.dat
grep "norm" out/diffusion-test.dat | cut -d '=' -f 2 > out/norm-test.dat
