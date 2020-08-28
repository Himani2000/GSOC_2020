import cv2
import face_recognition 
import os
import numpy as np
import pickle

# to run this code in the singularity perform
#singularity exec -B `pwd` biometric_deploy/bio-metric.img python3 face_features.py

if __name__ == '__main__':
    dirpath='ideology_image_dataset'
    files=list(os.listdir(dirpath))
    file_paths=[os.path.join(dirpath,file) for file in files]
    print("Total files ->{} ".format(len(file_paths)))
    file_paths.sort()
    # 1095 has no faces present 
    #file_paths=file_paths[12:14]
    total_face_embeddings=[]
    try:
        for i,image_path in enumerate(file_paths):
            #print(image_path)
                image=cv2.imread(image_path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, boxes)
                print("For the image path {} the face founds are {} ".format(image_path,len(encodings)))
                #print(boxes)
                if(len(encodings)==0):
                    print("No face detected... ")
                    face_embedding=[0 for i in range(128)]
                elif(len(encodings)==1):
                    print("One face detected... ")
                    face_embedding=encodings[0]
                    print(face_embedding)
                elif(len(encodings)>=2):
                    print("Multiple face detected... ")
                    max_area=0
                    for box,encoding in zip(boxes,encodings):
                        top=box[0]
                        right=box[1]
                        bottom=box[2]
                        left=box[3]
                        side1=abs(bottom-top)+1   
                        side2=abs(right-left)+1
                        area=side1*side2
                        print("The area is {} for the box {} ".format(area,box))
                        if(area>max_area):
                            max_area=area
                            max_box=box
                            face_embedding=encoding
                    #print("The maximum area face is {} that have boxes{} and the face embedding is {}".format(max_area,max_box,face_embedding))

                # here now i have the face embedding which is largest in the case if there are 
                #multiple faces present in it
                data=[{'image-path':image_path,'face-embedding':face_embedding}]
                total_face_embeddings.append(data)
    
    except Exception as e:
            print("EXCEPTION CAUSED >>> ",e)
    
    finally:
            f = open("Face_Embeddings.pickle", "wb")
            f.write(pickle.dumps(total_face_embeddings))
            f.close()
            print("DATA WRITTEN ... ")


