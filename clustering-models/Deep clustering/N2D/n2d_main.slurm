#!/bin/bash
#SBATCH -N 1 
#SBATCH --mem=70gb
#SBATCH --time=0-22:00:00 # 10 hrs  minutes 
#SBATCH --output=mygpu_n2d_main.stdout
#SBATCH --mail-user=hxn147@case.edu 
#SBATCH --mail-type=ALL 
#SBATCH --job-name="autoencoder_running not crop " 
# Put commands for executing job below this line 
# This example is loading the default Python module and then 
# writing out the version of Python 
module load gcc/6.3.0  openmpi/2.0.1
module load python/3.6.6

echo "this is the main n2d script used for calling"

#sbatch n2d_extract.slrum

echo "Calling the python script"

python n2d_main.py

#sbatch results_generation.sh

echo "Complete"



