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

#python autoencoder_feature_extraction.py ideology_image_dataset image_features_autoencoder_helper autoencoder_features ideology

#python autoencoder_feature_extraction.py ideology_crop_image_dataset image_features_crop_autoencoder_helper autoencoder_features_crop ideology_crop

echo "extracting the autoencoder features for the person dataset"

python autoencoder_feature_extraction.py ideology_person_dataset ideology_person_dataset_features_conventional ideology_person_dataset_features_autoencoder ideology


echo "extracting the features for the face dataset"

python autoencoder_feature_extraction.py ideology_face_dataset ideology_face_dataset_features_conventional ideology_face_dataset_features_autoencoder ideology

