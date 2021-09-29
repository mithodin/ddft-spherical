#!/bin/bash

outfolder=$1
if [[ "$outfolder" == "" ]];then
    outfolder=.
fi

if [[ ! -d "$outfolder" ]];then
    echo folder "'"$outfolder"'" does not exist
    exit 1
fi

normfile="$outfolder"/norm.dat
echo Norm: $normfile
touch "$normfile"
echo "#1_t 2_norm_self 3_norm_dist" > "$normfile"

dr=0.0078125
skip=false

while read line
do
    if [[ "$line" == *"# t ="* ]];then

        r=0

        time="${line:6}"
        time=$(printf %.6f $time)
        timelist="$timelist $time"

        outfile="$outfolder"/diffusion-t"$time".dat
        echo Diffusion: $outfile

        if [[ -f "$outfile" ]];then
            skip=true
            continue
        else
            skip=false
        fi

        touch "$outfile"
        echo "#1_r 2_rho_self 3_rho_dist 4_j_self 5_j_dist" > "$outfile"
        
    elif  [[ "$line" == *"# norm ="* ]];then

        norm="${line:9}"

        echo "$time $norm" >> "$normfile"

    elif $skip || [[ "$line" == *"#"* ]] || [[ "$line" == "" ]];then
        continue
    else
        echo "$r $line" >> "$outfile"
        r=$(echo $r+$dr|bc -l)
    fi
done < out/diffusion-test.dat
