#!/bin/bash
#SBATCH -N 1 
#SBATCH --mem=70gb
#SBATCH --time=0-22:00:00 # 10 hrs  minutes 
#SBATCH --output=mygpu.stdout 
#SBATCH --mail-user=hxn147@case.edu 
#SBATCH --mail-type=ALL 
#SBATCH --job-name="dec model "
#SBATCH --partition=gpu
#SBATCH -C gpuk40
#SBATCH --gres=gpu:1  

# Put commands for executing job below this line 
# This example is loading the default Python module and then 
# writing out the version of Python 

module spider tensorflow/1.4.0-py3
module load intel/17 openmpi/2.0.1 
module load tensorflow/1.4.0-py3

python image_feature_extraction_with_pretrainedmodels.py  ideology_person_dataset  ideology_person_dataset_features


echo "completed job "