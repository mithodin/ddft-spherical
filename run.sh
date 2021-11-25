#!/bin/sh
mkdir -p out
python ./main.py > out/diffusion.dat "$([ -z "$DDFT_CONFIG" ] && echo "config.jsonc" || echo "$DDFT_CONFIG")"
grep "norm_self" out/diffusion.dat | cut -d '=' -f 2 > out/norm-self.dat
grep "norm_dist" out/diffusion.dat | cut -d '=' -f 2 > out/norm-dist.dat
