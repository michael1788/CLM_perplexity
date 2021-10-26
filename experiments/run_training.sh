#!/bin/bash

CONFIG_FILE=$1

if (( $# != 1 )); then
    echo "You have to provide 1 argument; the path to the config file"
else

# declare an array variable
declare -a arrcid=("CHEMBL1836" "CHEMBL1945" "CHEMBL1983" "CHEMBL202" "CHEMBL3522" "CHEMBL4029" "CHEMBL5073" "CHEMBL5137" "CHEMBL5408" "CHEMBL5608")
declare -a arrsize=("_5" "_10" "_20" "_40")

for f in "${arrcid[@]}";
    do for s in "${arrsize[@]}";
        do python do_training.py --configfile $CONFIG_FILE --repeat 0 --name_data "$f$s"
    done
done

fi