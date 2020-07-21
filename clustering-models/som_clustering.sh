#!/bin/bash 

#echo "running  the som for the pretrained features"

#python som_clustering.py ideology_image_dataset Image_features som_clustering_results

#echo "running for the som for non pretrained features"

#python som_clustering.py ideology_image_dataset image_features_15july som_clustering_results

echo "Running for generating the text and the pdf files with filter"
python image_results_utils.py ideology_image_dataset som_clustering_results/ image_som_results_text_pdf yes


#echo "running  the som for the pretrained features crop "

#python som_clustering.py ideology_crop_image_dataset Image_features_crop som_clustering_results_crop

#echo "running for the som for non pretrained features"

#python som_clustering.py ideology_crop_image_dataset image_features_15july_crop som_clustering_results_crop

#echo "Running for generating the text and the pdf files with filter"
#python image_results_utils.py ideology_image_dataset som_clustering_results/ image_som_results_text_pdf yes