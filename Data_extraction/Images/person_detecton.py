import cv2
import imutils

#from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


#trying the yolo method 

def yolo(image_path,threshold_confidence=0.5,threshold=0.3,visualize=True):
    #threshold_confidence=0.5
    #threshold=0.3
    
    label_file_path='coco.names'
    weights_path='yolov3.weights'
    config_path='yolov3.cfg'
    
    with open(label_file_path,'r')as f:
        labels=f.read().split('\n')
    #print("Total labels {} present".format(len(labels)))
    
    person_detected=False
    person_coordinates=[]
    
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    image=cv2.imread(image_path)
    
    if(visualize):
        fig = plt.figure()
        plt.figure(figsize=(20,10)) 
        plt.subplot(3, 1, 1)
        plt.title("Original Image")
        plt.imshow(image)
    
    (h,w)=image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores=detection[5:]
            class_label_id=np.argmax(scores)
            confidence=scores[class_label_id]
            if(confidence>threshold_confidence):
                box=detection[0:4]*np.array([w,h,w,h])
                (centerx,centery,width,height)=box.astype("int")
                x=int(centerx-(width/2))
                y=int(centery-(height/2))
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(class_label_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences,threshold_confidence,threshold)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if(labels[classIDs[i]]=="person"):
                person_detected=True
                #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
                person_coordinates.append([x,y,x+w,y+h])
                
                
            #print(labels[classIDs[i]])
    
    if(visualize):
        plt.subplot(3,1,2)
        plt.title("Person detection")
        plt.imshow(image)
    
    return person_coordinates,person_detected

def Sort(sub_li): 
    return(sorted(sub_li, key = lambda x: x[0]))   


def updateImageDimensions(all_images,height,width):
    dim=(width,height)
    updated_images=[]
    for image in all_images:
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        updated_images.append(resized)
    return updated_images
        
def concatenateImages(updated_images):
    if(len(updated_images)>=2):
        image_concatenated=updated_images[0]
        for image in updated_images[1:]:
            image_concatenated = np.concatenate((image_concatenated, image), axis=1)
    
        return image_concatenated
    else:
        
        pass
                  
if __name__ == '__main__':

    files=list(os.listdir('ideology_image_dataset/'))
    filepaths=[os.path.join('ideology_image_dataset/',file) for file in files]
    print("Total files",len(filepaths))
    for imagePath in filepaths:
        image=cv2.imread(imagePath)
        person_coordinates,person_detected=yolo(imagePath,0.8,0.3)
        person_coordinates=Sort(person_coordinates)
        filename=os.path.join("ideology_person_dataset",imagePath.split("/")[1])
        print(filename)
        print("Person coordinates ",person_coordinates)
        #fig = plt.figure()
        if(person_detected):
            if(len(person_coordinates)>=2):
                no_person_detected=len(person_coordinates)
                all_images=[]
                max_w=0
                max_h=0
                for person_coordinate in person_coordinates:
                    startX,startY,endX,endY=person_coordinate[0],person_coordinate[1],person_coordinate[2],person_coordinate[3]
                    print(startX,endX,startY,endY)
                    if(startX>=0 and startY>=0 and endX>=0 and endY>=0):
                        new_image=image[startY:endY,startX:endX]
                        all_images.append(new_image)
                        h,w=new_image.shape[:2]
                       # print("new image",new_image.shape)
                        max_w=max(max_w,w)
                        max_h=max(max_h,h)

                print(max_w,max_h)
                updated_images=updateImageDimensions(all_images,max_h,max_w)
                #print(updated_images[0].shape,updated_images[1].shape)
                image_concatenated=concatenateImages(updated_images)
                #print(image_concatenated.shape)
                #plt.title("No of person detected %i"%no_person_detected)
                #plt.imshow(image_concatenated)
                try:
                    cv2.imwrite(filename,image_concatenated)

                except:
                    cv2.imwrite(filename,image)
                    pass

            else:
                no_person_detected=len(person_coordinates)
                person_coordinate=person_coordinates[0]
                startX,startY,endX,endY=person_coordinate[0],person_coordinate[1],person_coordinate[2],person_coordinate[3]
                if(startX>=0 and startY>=0 and endX>=0 and endY>=0):
                    new_image=image[startY:endY,startX:endX]
                #plt.title("No of face detected %i"%no_person_detected)
                #plt.imshow(new_image)
                try:
                    cv2.imwrite(filename,new_image)
                except:
                    cv2.imwrite(filename,image)
                    pass


        else:
            no_person_detected=0
            h,w=image.shape[0:2]
            x=0
            y=0
            h=h*0.65
            h=int(h)
            image=image[y: y+h, x: x+w] 
            cv2.imwrite(filename,image)


        
        



