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

def checkFilter(predicted_labels,files_path,filter_dict,text_file):
    unique_labels=set(predicted_labels)
    text_file.write(f"Total clusters found -->{len(unique_labels)}\n")
    trump_rnc=['clip_1014.jpg', 'clip_1153.jpg', 'clip_1161.jpg', 'clip_1162.jpg', 'clip_1163.jpg', 'clip_1164.jpg', 'clip_1167.jpg', 'clip_1168.jpg', 'clip_1169.jpg', 'clip_1170.jpg', 'clip_1172.jpg', 'clip_1173.jpg', 'clip_1175.jpg', 'clip_1177.jpg', 'clip_1178.jpg', 'clip_1179.jpg', 'clip_1180.jpg', 'clip_1182.jpg', 'clip_1183.jpg', 'clip_1185.jpg', 'clip_1186.jpg', 'clip_1187.jpg', 'clip_1188.jpg', 'clip_1189.jpg', 'clip_1190.jpg', 'clip_1191.jpg', 'clip_1192.jpg', 'clip_1193.jpg', 'clip_1196.jpg', 'clip_1197.jpg', 'clip_1198.jpg', 'clip_1200.jpg', 'clip_1201.jpg', 'clip_1202.jpg', 'clip_1203.jpg', 'clip_1204.jpg', 'clip_1205.jpg', 'clip_1206.jpg', 'clip_1207.jpg', 'clip_1208.jpg', 'clip_1209.jpg', 'clip_1210.jpg', 'clip_1211.jpg', 'clip_1212.jpg', 'clip_486.jpg', 'clip_890.jpg']
    flyn=['clip_155.jpg', 'clip_156.jpg', 'clip_191.jpg', 'clip_196.jpg', 'clip_197.jpg', 'clip_215.jpg', 'clip_218.jpg', 'clip_219.jpg', 'clip_223.jpg', 'clip_270.jpg', 'clip_275.jpg', 'clip_277.jpg', 'clip_278.jpg', 'clip_285.jpg', 'clip_290.jpg', 'clip_291.jpg', 'clip_294.jpg', 'clip_295.jpg', 'clip_296.jpg', 'clip_301.jpg', 'clip_309.jpg', 'clip_310.jpg', 'clip_311.jpg', 'clip_312.jpg', 'clip_317.jpg', 'clip_318.jpg', 'clip_323.jpg', 'clip_324.jpg', 'clip_326.jpg', 'clip_327.jpg', 'clip_329.jpg', 'clip_330.jpg', 'clip_332.jpg', 'clip_333.jpg', 'clip_336.jpg', 'clip_337.jpg', 'clip_344.jpg', 'clip_345.jpg', 'clip_346.jpg', 'clip_348.jpg', 'clip_349.jpg', 'clip_351.jpg', 'clip_352.jpg', 'clip_359.jpg', 'clip_360.jpg', 'clip_373.jpg', 'clip_374.jpg', 'clip_376.jpg', 'clip_38.jpg', 'clip_381.jpg', 'clip_382.jpg', 'clip_386.jpg', 'clip_388.jpg', 'clip_389.jpg', 'clip_39.jpg', 'clip_391.jpg', 'clip_392.jpg', 'clip_40.jpg', 'clip_400.jpg', 'clip_404.jpg', 'clip_405.jpg', 'clip_408.jpg', 'clip_409.jpg', 'clip_41.jpg', 'clip_413.jpg', 'clip_414.jpg', 'clip_42.jpg', 'clip_6.jpg', 'clip_7.jpg']
    khewsis=['clip_1925.jpg', 'clip_2494.jpg', 'clip_2495.jpg', 'clip_2497.jpg', 'clip_2500.jpg', 'clip_2501.jpg', 'clip_2502.jpg', 'clip_2503.jpg', 'clip_2504.jpg', 'clip_2505.jpg', 'clip_2506.jpg', 'clip_2507.jpg', 'clip_2509.jpg', 'clip_2510.jpg', 'clip_2511.jpg', 'clip_2513.jpg', 'clip_2514.jpg', 'clip_2515.jpg', 'clip_2517.jpg']

    if(len(unique_labels)>2200):
        return False  ## we may aslo not need this 
    
    elif(len(unique_labels)<30):
        return False # we may not need this 
    
    #print("length of unique labels ",len(unique_labels))
    
    for label in list(unique_labels):
        label_indexs= np.where(predicted_labels==label)[0]
        
        files=[file.split('/')[1] for file in files_path]
        cluster_files=[files[i] for i in label_indexs]
       # print(cluster_files)
        
        if(len(set(cluster_files)&set(trump_rnc))!=0):
           # print("writing to a text file")
            cluster_num=filter_dict['trumprnc']+1
            text_file.write(f'Number of images in trump cluster-> {cluster_num}-> {len(set(cluster_files)&set(trump_rnc))} \n')
            filter_dict['trumprnc']+=1
        
    
        if(len(set(cluster_files)&set(flyn))!=0):
            #print("writing to a text file")
            cluster_num=filter_dict['flyn']+1
            text_file.write(f'Number of images in flynn cluster-> {cluster_num}-> {len(set(cluster_files)&set(flyn))} \n')
            filter_dict['flyn']+=1


        if(len(set(cluster_files)&set(khewsis))!=0):
            #print("writing to a text file")
            cluster_num=filter_dict['khewsis']+1
            text_file.write(f'Number of images in khewsis cluster->{cluster_num}-> {len(set(cluster_files)&set(khewsis))} \n')
            filter_dict['khewsis']+=1
            
    
    text_file.write(f"Total clusters for trump rnc is {filter_dict['trumprnc']}\n")
    text_file.write(f"Total clusters for flynn is {filter_dict['flyn']}\n")
    text_file.write(f"Total clusters for khewsis rnc is {filter_dict['khewsis']}\n")
    
    text_file.close()
    if(filter_dict['trumprnc']>4):
        return False
    
    if(filter_dict['flyn']>5):
        return False
    
    if(filter_dict['khewsis']>3):
        return False
    
    
    return True
    


def showClustering(predicted_labels,label,pdf,algorithm,files_path):
    label_indexs= np.where(predicted_labels==label)[0]
    print("CLUSTER--> ",label,"TOTAL IMAGES--> ",len(label_indexs))
    
    

    if(len(label_indexs)>=500):
        fig=plt.figure(figsize=(10, 300))
        
        
    elif(len(label_indexs)>100 and len(label_indexs)<500):
        fig=plt.figure(figsize=(10, 40))
    elif(len(label_indexs)>=50 and len(label_indexs)<100):
        fig=plt.figure(figsize=(10, 10))
        
    elif(len(label_indexs)>=20 and len(label_indexs)<50):
        fig=plt.figure(figsize=(10, 3))
    
    elif(len(label_indexs)>=0 and len(label_indexs)<20):
        fig=plt.figure(figsize=(10, 2))
    
    plt.title(f'The cluster -> {label} and total images->{len(label_indexs)}')
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(hspace = 0, wspace = 0)

    for i,index in enumerate(label_indexs):
       
        
        image = cv2.imread(files_path[index])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        columns = 10
        rows = np.ceil(len(label_indexs)/float(columns))
        
        fig.add_subplot(rows,columns, i+1)
        plt.axis("off")
       
        plt.imshow(image)
        
    
   
    pdf.savefig()
    plt.close(fig)
    plt.clf()
    plt.cla()
    
    
if __name__ == '__main__':
	my_parser = argparse.ArgumentParser(description='List the content of a folder')

	my_parser.add_argument('image_path',
	                       metavar='path',
	                       type=str,
	                       help='the path to the image directory')
	my_parser.add_argument('predicted_labels_path',
	                       metavar='path',
	                       type=str,
	                       help='the path to the predicted_labels')

	my_parser.add_argument('results_path',
	                       metavar='path',
	                       type=str,
	                       help='the path where the results should be stored')

	my_parser.add_argument('filter',
	                       metavar='bool',
	                       type=str,
	                       help='the path where the results should be stored')

	args = my_parser.parse_args()

	image_path=args.image_path
	predicted_labels_path=args.predicted_labels_path
	results_path=args.results_path
	filter_=args.filter
	#if(os.path.isfile(results_path)==False):
	if not os.path.exists(results_path):
		print("No directory exists of results")
		os.makedirs(os.path.join(os.getcwd(),results_path,"text_files"))
		os.makedirs(os.path.join(os.getcwd(),results_path,"pdf_files"))

		results_path_text_files=os.path.join(os.getcwd(),results_path,"text_files")		
		results_path_pdf_files=os.path.join(os.getcwd(),results_path,"pdf_files")

	else:
		pass	

	#print(results_path,results_path_text_files,results_path_pdf_files)
	files_path=load_filepaths(image_path)

	for file in list(os.listdir(predicted_labels_path)):
		if file.endswith('.npy'):
			print(file)
			filter_dict={"flyn":0,"trumprnc":0,"khewsis":0}
			predicted_labels=np.load(f'{predicted_labels_path}{file}')
			unique_labels=set(predicted_labels)
			filename="".join(file.split('.npy'))
			print("total clusters ",len(unique_labels))
			#filename=f'{filename}.txt'
			text_file=open(os.path.join(results_path_text_files,f'{filename}.txt'),'w')
			text_file.write(f"Writing for the file -->  {filename} \n")

			pdf_save=checkFilter(predicted_labels,files_path,filter_dict,text_file)

			text_file.close()
			
			if(pdf_save):
				with PdfPages(os.path.join(results_path_pdf_files,f'{filename}.pdf')) as pdf:
					for label in list(unique_labels):
						if(label!=-1):

							showClustering(predicted_labels,label,pdf,filename,files_path)







	#print(image_path,predicted_labels_path,results_path,filter_.lower())
