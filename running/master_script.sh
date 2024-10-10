#!/bin/bash

parametrization=(pp cs) #pp cs
order=(n3lo-mod)           ###########CAUTION: important alterations on base.py to make NLO bands run OFF, mock gaussian band ceft ON
core_start=(1.1 1.5) # 1.1 or 1.5
case=(prior)

#i=0
for p in ${parametrization[@]}
do
    for o in ${order[@]}
    do
        for c in ${core_start[@]}
        do
            for d in ${case[@]}
            do
                #echo "$i"
                echo "$p $o $c $d"
                sbatch --account=project02319 --ntasks=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem-per-cpu=1000 --time=24:00:00 --output=slurm/%J_out.txt --error=slurm/%J_err.txt --wrap="./ind_script.sh $p $o $c $d" # '/home/mm12wyxy/neost-0.10/DANEoST/repro/"$d"/"$p"/"$o"_m2/15' "  #'PP-prior-ceft-Keller-N3LO-n0-1.5-cut-ID34549-' " # ${id[$i]}"
                #((i=i+1))
            done
        done
    done
done

