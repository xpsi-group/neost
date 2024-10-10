#!/bin/bash

# -r: use reproduced results instead of published
# -t: output to tmp.pdf instead of normal file name
# -s: use scatter plot instead of default
# -n: use numpy histogram 2D plot instead of seaborn KDE
# -a: plot all datasets rather than just our chosen scenarios
# -o: chiral order, N2LO or N3LO (defaults to N3LO)
# -d: select which datasets to plot
# -bo: plot both orders

optional_args=$1

if [ "$optional_args" == "-r" ]
then
	printf "Using reproduced results.\n"
else
	printf "Using published results.\n"
	printf "Use flag -r with this script to plot reproduced (as opposed to published) results.\n"
fi

mkdir -p fig

printf "\nFig. 1\n"
cmd="python plot_routines/cEFT_band.py" # -t
printf "$cmd\n"
$cmd

printf "\nFig. 2\n"
cmd="python plot_routines/MR_priors.py $optional_args" # -r / -n / -s / -t
printf "$cmd\n"
$cmd

printf "\nFig. 3\n"
cmd="python plot_routines/pressure_energydensity_priors.py $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 4\n"
cmd="python plot_routines/data_likelihood.py" # -a / -t
printf "$cmd\n"
$cmd

printf "\nFig. 5\n"
cmd="python plot_routines/MR_baseline_new.py $optional_args" # -r / -n / -s / -t
printf "$cmd\n"
$cmd

printf "\nFig. 6\n"
cmd="python plot_routines/pressure_energydensity_posteriors.py $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 7\n"
cmd="python plot_routines/MR_new_heatmap.py $optional_args" # -r / -t / -s
printf "$cmd\n"
$cmd

printf "\nFig. 8\n"
cmd="python plot_routines/pressure_histograms.py $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 9\n"
cmd="python plot_routines/R2vsR14.py $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 10\n"
cmd="python plot_routines/DeltaRvsMTOV.py -d New $optional_args" # -r / -t / -bo
printf "$cmd\n"
$cmd

printf "\nFig. 11\n"
cmd="python plot_routines/data_likelihood.py -a" # -t
printf "$cmd\n"
$cmd

printf "\nFig. 12\n"
cmd="python plot_routines/MR_posteriors.py -d New $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 13\n"
cmd="python plot_routines/pressure_histograms.py -o N2LO $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 14\n"
cmd="python plot_routines/DeltaRvsMTOV.py -d New -bo $optional_args" # -r / -t
printf "$cmd\n"
$cmd

printf "\nFig. 15\n"
cmd="python plot_routines/MR_new_new2_new3.py -s 15 $optional_args" # -r / -t / -o. '-s 15' (core start at 1.5n_0) can be changed to '-s 11'
printf "$cmd\n"
$cmd
