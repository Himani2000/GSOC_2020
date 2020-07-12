#!/bin/bash 

echo "Running all the clustering models"

echo "Running for the density based clustering algorithms"

python densityclustering.py ideology_crop_image_dataset Image_features_crop Image_results_crop


echo "Running for the hierarchical clustering algorithms"
python hierarchicalclustering.py ideology_crop_image_dataset Image_features_crop Image_results_crop

echo "Running for generating the text and the pdf files with filter"
python image_results_utils.py ideology_crop_image_dataset Image_results_crop/ Image_results_crop_text_pdf yes