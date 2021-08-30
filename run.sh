#!/bin/bash
if [ ! -f "vanhove.h5" ]; then
  wget "https://files.treffenstaedt.de/vanhove.h5" -O vanhove.h5
fi
mkdir -p out
python3 ./main.py > out/diffusion.dat
grep "norm" out/diffusion.dat | cut -d '=' -f 2 > out/norm.dat
