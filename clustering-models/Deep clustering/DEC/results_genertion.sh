#!/bin/bash 
#SBATCH -N 1 
#SBATCH --mem=70gb
#SBATCH --time=0-24:00:00 # 10 hrs  minutes 
#SBATCH --output=results_pdf.stdout 
#SBATCH --mail-user=hxn147@case.edu 
#SBATCH --mail-type=ALL 
#SBATCH --job-name="pfresults" 
# Put commands for executing job below this line 
# This example is loading the default Python module and then 
# writing out the version of Python 


module load gcc/6.3.0  openmpi/2.0.1
module load python/3.6.6



echo "SAVING THE RESULTS ........... "

echo "RESULTS GENERATION DBC  ............ "

python image_results_utils.py /mnt/rds/redhen/gallina/home/hxn147/ideology_image_dataset results2/ results_text_pdf yes 










