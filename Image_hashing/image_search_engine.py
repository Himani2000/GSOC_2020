from PIL import Image
import imagehash
import os
import argparse 
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import vptree
import cv2
from imutils import build_montages


def load_filepaths(image_dir):
    
    files=os.listdir(image_dir)
    files_path=[os.path.join(image_dir,file) for file in files ]
    
    return files_path

def hamming_distance(hash1,hash2):
    return abs(hash1-hash2)

def similarImages(hashfunc,files_path):
    similar_images={}
    for filepath in files_path:
        image_hash=hashfunc(Image.open(filepath))
        similar_images[image_hash]=similar_images.get(image_hash, []) + [filepath]
        
    return similar_images

def showMontage(image_paths):
    images=[]
    columns = 4
    rows = np.ceil(len(image_paths)/float(columns))
    if(rows>10):
        columns=columns+4
        
    print(columns,rows)
    for imagePath in image_paths:
        # load the image and update the list of images
        image = cv2.imread(imagePath)
        images.append(image)
    # construct the montages for the images
    montages = build_montages(images, (128, 196), (int(rows), int(columns)))
    #cv2.imwrite("monatge.png",montages)
    for montage in montages:
        #cv2.imshow("Montage", montage)
        cv2.imwrite("monatge.png",montage)
        #cv2.waitKey(0)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    my_parser.add_argument('image_path',metavar='path',type=str,help='the path to the image directory')
    
    my_parser.add_argument('hashtype',type=str,help='the hash function')
   
    my_parser.add_argument('threshold',type=int,help='the distance threshold for finding the similar images')
   
    my_parser.add_argument('query_image_path',metavar='path',type=str,help='the path to the query  image file')
   
    #my_parser.add_argument('query_image_name',type=str,help='the hash function')
    
    
    args = my_parser.parse_args()

    image_path=args.image_path
    hashtype=args.hashtype
    threshold=args.threshold
    query_image_path=args.query_image_path
    
    print("loading files path ")
    files_path=load_filepaths(image_path)
    
    
    
    if(hashtype=="dhash"):
        hashfunction=imagehash.dhash
        
    elif(hashtype=="ahash"):
        hashfunction=imagehash.average_hash
        
    elif(hashtype=="phash"):
        hashfunction=imagehash.phash
    
    elif(hashtype=="whash"):
        hashfunction=imagehash.whash
        
    print("computing the hash values for all the images ...")
    similar_images=similarImages(hashfunction,files_path)
    
    query_hash=hashfunction(Image.open(os.path.join(image_path,query_image_path)))
    
    
    print("contructing the tree ...")
    points = list(similar_images.keys())
    tree = vptree.VPTree(points, hamming_distance)
    
    results = tree.get_all_in_range(query_hash,threshold)
    
    image_paths=[]
    for d,h in results:
    #print(h)
        resultPaths = similar_images.get(h, [])
        image_paths.extend(resultPaths)
        
    print("Total images found :",len(image_paths))
    showMontage(image_paths)

    
        
    
    