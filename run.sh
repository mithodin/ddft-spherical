mkdir -p out
python3 ./main.py > out/diffusion.dat
grep "norm" out/diffusion.dat | cut -d '=' -f 2 > out/norm.dat
