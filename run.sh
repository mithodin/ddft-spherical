#!/bin/sh
mkdir -p out
python ./main.py > out/diffusion.dat "$([ -z "$DDFT_CONFIG" ] && echo "config.jsonc" || echo "$DDFT_CONFIG")"
grep "norm" out/diffusion.dat | cut -d '=' -f 2 > out/norm.dat
