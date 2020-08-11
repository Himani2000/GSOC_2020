#!/bin/bash 

#SBATCH -N 1 
#SBATCH --mem=90gb
#SBATCH --time=0-24:00:00 # 10 hrs  minutes 
#SBATCH --output=mygpu.stdout 
#SBATCH --mail-user=hxn147@case.edu 
#SBATCH --mail-type=ALL 
#SBATCH --job-name="autoencoderfeatures "
#SBATCH --partition=gpu
#SBATCH -C gpuk40
#SBATCH --gres=gpu:1  

module spider tensorflow/1.4.0-py3
module load intel/17 openmpi/2.0.1 
module load tensorflow/1.4.0-py3



echo "running the autoencoder script"

python autoencoder_feature_extraction.py image_dataset image_features autoencoder_features image-dataset-name

python autoencoder_feature_extraction.py image_dataset image_features autoencoder_features_crop image-dataset-name




