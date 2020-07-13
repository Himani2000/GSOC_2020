from PIL import Image
import imagehash
import os
import argparse 
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import cv2
from matplotlib.backends.backend_pdf import PdfPages



def load_filepaths(image_dir):
    
    files=os.listdir(image_dir)
    files_path=[os.path.join(image_dir,file) for file in files ]
    
    return files_path



def saveResult(images_path,pdf,count):
    fig=plt.figure(figsize=(10, 2))
    plt.title(f"The similar  {count} th similar hash images")
    plt.gca().set_axis_off()
    plt.subplots_adjust(hspace = 0, wspace = 0)
    #print("inside the save result len is ,",len(images_path))
    #print(images_path)
    for i,imagepath in enumerate(images_path):
     #   print(imagepath)
        image = cv2.imread(imagepath)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        columns = 2
        rows = np.ceil(len(images_path)/float(columns))

        fig.add_subplot(rows,columns, i+1)
        plt.axis("off")

        plt.imshow(image)
        
    
   
    pdf.savefig()
    plt.close(fig)
    plt.clf()
    plt.cla()
    
    
def similarImages(hashfunc,files_path):
    similar_images={}
    for filepath in files_path:
        image_hash=hashfunc(Image.open(filepath))
        similar_images[image_hash]=similar_images.get(image_hash, []) + [filepath]
        
    return similar_images


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    my_parser.add_argument('image_path',metavar='path',type=str,help='the path to the image directory')
    

    args = my_parser.parse_args()

    image_path=args.image_path
    #image_path='ideology_image_dataset'
    hashfunclist=[imagehash.average_hash,imagehash.phash,imagehash.dhash,imagehash.whash,imagehash.colorhash]
    hashfuncs=['avergae_hash','p_hash','d_hash','w_hash','color_hash']
    
    files_path=load_filepaths(image_path)
    for i,hashfunc in enumerate(hashfunclist):
        print(hashfuncs[i])
        similar_images=similarImages(hashfunc,files_path)
        print("computed hash..")
        filename=hashfuncs[i]
        with PdfPages(f'{image_path}_{filename}.pdf') as pdf:
            count=0
            for hashes in list(similar_images.keys()):
                
                if(len(similar_images[hashes])>=2):
                    print(count,len(similar_images[hashes]))
                    saveResult(similar_images[hashes],pdf,count+1)
                    count+=1
            
        
        
    