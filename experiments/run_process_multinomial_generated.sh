#!/bin/bash

CONFIG_FILE=$1
RANGESTART=$2
STEP=$3
RANGEEND=$4

correction=""
NAME_CONFIG_FILE=${CONFIG_FILE/.ini/$correction}
NAME_CONFIG_FILE=${NAME_CONFIG_FILE/configfiles/$correction}
OUTPUT="output"

if (( $# != 4 )); then
    echo "You have to provide 4 arguments; the path to the config file, the epoch range you wanna sample from (start, step, end)"
else

# declare an array variable
declare -a arrcid=("CHEMBL1836" "CHEMBL1945" "CHEMBL1983" "CHEMBL202" "CHEMBL3522" "CHEMBL4029" "CHEMBL5073" "CHEMBL5137" "CHEMBL5408" "CHEMBL5608")
declare -a arrsize=("_5" "_10" "_20" "_40")

for f in "${arrcid[@]}";
    do for s in "${arrsize[@]}";
        do for m in $(seq $RANGESTART $STEP $RANGEEND);
            do python do_process_multinomial_generated.py --configfile $CONFIG_FILE --repeat 0 --name_data "$f$s" --epoch $m
        done
    done
done

fi