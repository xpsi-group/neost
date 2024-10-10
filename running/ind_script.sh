# This is to activate the neost environment.
# Needs to be adapted to each user's situation.
if [ "$USER" = "svensson" ]; then #not sure what's Isak login name here
        source senv.sh neost-0.10
elif [ "$USER" = "mm12wyxy" ]; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda activate neost-0.10
else
    	echo "New user, you need to specify how to activate your neost environent."
        exit 0
fi

fname="sample_mod_gaussian.py"              ### NLO modifications in base.py OFF; Likelihood M_TOV>2.0 M_sun OFF; Mock gaussian ceft band ON
echo "Running $fname"
time python -u $fname -$1 -o $2 -s $3 -c $4 #-d $5 #-name $6 # 100 #-id $SLURM_JOBID


