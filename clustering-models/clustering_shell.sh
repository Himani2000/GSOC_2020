#!/bin/bash 

echo "Running all the clustering models"

#echo "Running for the density based clustering algorithms"

#python densityclustering.py ideology_image_dataset image_features_15july image_results_15july


#echo "Running for the hierarchical clustering algorithms"
#python hierarchicalclustering.py ideology_image_dataset image_features_15july image_results_15july


#echo "Running for the density -crop"
#python densityclustering.py ideology_crop_image_dataset image_features_15july_crop image_results_15july_crop


#echo "Running for the hierarchical clustering algorithms -- crop "
#python hierarchicalclustering.py ideology_crop_image_dataset image_features_15july_crop image_results_15july_crop

echo "Running for generating the text and the pdf files with filter"
python image_results_utils.py ideology_image_dataset image_results_15july/ image_results_15july_text_pdf4 yes

echo "Running for generating the text and the pdf files with filter-- crop "
python image_results_utils.py ideology_crop_image_dataset image_results_15july_crop/ image_results_15july_crop_text_pdf4 yes