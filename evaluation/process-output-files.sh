#!/bin/bash

outfolder=$1
if [[ "$outfolder" == "" ]];then
    outfolder=.
fi

if [[ ! -d "$outfolder" ]];then
    echo folder "'"$outfolder"'" does not exist
    exit 1
fi

dr=0.0078125
n=4096

infilediffusion=out/diffusion.dat

normfile="$outfolder"/norm.dat
echo Norm: $normfile
touch "$normfile"
echo "#1_t 2_norm_self 3_norm_dist" > "$normfile"

while read line
do
    if [[ "$line" == *"# t ="* ]];then

        time="${line:6}"
        time=$(printf %.6f $time)
        timelist="$timelist $time"

    elif  [[ "$line" == *"# norm ="* ]];then

        norm="${line:9}"

        echo "$time $norm" >> "$normfile"
    fi
done < $infilediffusion

i=0
for time in $timelist
do
    outfile="$outfolder"/diffusion-t"$time".dat
    
    i=$((i+1))
    lastline=$((i*(n+4)-2))

    echo Diffusion: $i $outfile

    echo "2_rho_self 3_rho_dist 4_j_self 5_j_dist 6_D_rho_shell_self 7_D_rho_shell_dist" > tmp

    head -n $lastline $infilediffusion|tail -n $n >> tmp

    if [[ -f "$outfile" ]];then
        rm "$outfile"
    fi
    touch "$outfile"
    paste out/radius.dat tmp >> "$outfile"
done

rm tmp

