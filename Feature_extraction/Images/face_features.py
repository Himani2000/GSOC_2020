import cv2
import face_recognition 
import os
import numpy as np



if __name__ == '__main__':
    dirpath='ideology_image_dataset'
    files=list(os.listdir(dirpath))
    file_paths=[os.path.join(dirpath,file) for file in files]
    print("Total files ->{} ".format(len(file_paths)))
    file_paths.sort()
    #print(file_paths[0:5])
    #print(file_paths[101])

    total_face_embeddings=[]
    total_average_face_embeddings=[]
    try:
        for i,image_path in enumerate(file_paths):
            image=cv2.imread(image_path)
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(image,model="cnn")
            encodings = face_recognition.face_encodings(image, boxes)
            print(" for the {} Total boxes {} and total encodings {} for the image {}".format(i+1,len(boxes),len(encodings),image_path))
            if(len(encodings)>=1):
                face_embedding=encodings[0]
                total_face_encoding=len(encodings)
                if(len(encodings)>=2):
                    for encoding in encodings[1:]:
                        face_embedding=[a+b for a,b in zip(face_embedding,encoding)]


                    average_face_embedding=[a/total_face_encoding for a in face_embedding]
                else:
                    average_face_embedding=face_embedding
                #print(face_embedding[0:5])
                #print(average_face_embedding[0:5])

            else:
                #print("No face detected ")
                face_embedding=[0 for i in range(128)]
                average_face_embedding=face_embedding
                #print(face_embedding[0:5])
                #print(average_face_embedding[0:5])

            total_face_embeddings.append(face_embedding)
            total_average_face_embeddings.append(average_face_embedding)

    except Exception as e:
        pass
    finally:
        total_face_embeddings = np.asarray(total_face_embeddings)
        total_average_face_embeddings=np.asarray(total_average_face_embeddings)
        print(len(total_face_embeddings))
        print(len(total_average_face_embeddings))
            #print("Length of the face face_embedding",len(face_embedding))

        #total_face_embeddings=np.array([total_face_embeddings])
        #total_average_face_embeddings=np.array([total_average_face_embeddings])

        print(total_average_face_embeddings.shape)
        print(total_face_embeddings.shape)

        np.save("Total_face_embeddings.npy",total_face_embeddings)
        np.save("Total_average_face_embeddings.npy",total_average_face_embeddings)


        print("Done..")

