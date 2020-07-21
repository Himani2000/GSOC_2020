#!/bin/bash 

echo "Running all the clustering models"

echo "Running for the density based clustering algorithms"

python image_feature_extraction_without_pretrainedmodels.py  ideology_image_dataset image_features_15july ideology

python image_feature_extraction_without_pretrainedmodels.py  ideology_crop_image_dataset image_features_15july crop_ideology