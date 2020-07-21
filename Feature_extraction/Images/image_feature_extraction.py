import numpy as np
import cv2
import argparse 
import os
from matplotlib import pyplot as plt
import pandas as pd
from skimage.feature import hog



def loadFilePaths(image_directory):
    
    files=os.listdir(image_directory)
    files_path=[os.path.join(image_directory,file) for file in files ]
    return files_path

def saveFeatures(features,modelname,filename,dirname):
    saved_filename=filename+'_'+modelname
    saved_filename=os.path.join(dirname,saved_filename)

    print("saving",saved_filename+'.npy')
    np.save(saved_filename+'.npy',features)

#getting the gray scale features for the images 
def getGrayScaleFeatures(filelist): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img=cv2.imread(imagepath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(img_gray,(3,3),0)
     
        width = 120
        height = 120
        dim = (width, height)
        
        resized = cv2.resize(img_gaussian, dim, interpolation = cv2.INTER_AREA)
        
        featurelist.append(resized.flatten())
        

    featurelist = np.asarray(featurelist)
    return featurelist

def getCannyFeatures(filelist): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img=cv2.imread(imagepath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(img_gray,(3,3),0)
     
        width = 120
        height = 120
        dim = (width, height)
        
        resized = cv2.resize(img_gaussian, dim, interpolation = cv2.INTER_AREA)
        
        img_canny = cv2.Canny(resized,100,200)
        featurelist.append(img_canny.flatten())
        

    featurelist = np.asarray(featurelist)
    return featurelist

def getSobelyFeatures(filelist): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img=cv2.imread(imagepath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(img_gray,(3,3),0)
     
        width = 120
        height = 120
        dim = (width, height)
        
        resized = cv2.resize(img_gaussian, dim, interpolation = cv2.INTER_AREA)
        
        img_sobel =cv2.Sobel(resized,cv2.CV_8U,0,1,ksize=5)
        featurelist.append(img_sobel.flatten())
        

    featurelist = np.asarray(featurelist)
    return featurelist


def getPrewittyFeatures(filelist): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img=cv2.imread(imagepath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(img_gray,(3,3),0)
     
        width = 120
        height = 120
        dim = (width, height)
        
        resized = cv2.resize(img_gaussian, dim, interpolation = cv2.INTER_AREA)
        kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        
       
        img_prewitty = cv2.filter2D(resized, -1, kernely)
        
        
        featurelist.append(img_prewitty.flatten())
        

    featurelist = np.asarray(featurelist)
    return featurelist



def getHogFeatures(filelist): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img=cv2.imread(imagepath)
        
     
        width = 164
        height = 128
        dim = (width, height)
        
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
       
        fd, hog_image = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,multichannel=True)
        
        
        featurelist.append(fd.flatten())
        

    featurelist = np.asarray(featurelist)
    return featurelist

def getFastFeatures(filelist):
	filelist.sort()
	featurelist = []
	#featurelist=np.array([])
	for i, imagepath in enumerate(filelist):
		print(" Status: %s / %s" %(i, len(filelist)), end="\r")
		img=cv2.imread(imagepath,0)
		fast = cv2.FastFeatureDetector_create()
		# find and draw the keypoints
		width = 164
		height = 128
		dim = (width, height)

		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

		kp = fast.detect(resized,None)
		br = cv2.BRISK_create();
		kp, des = br.compute(resized,  kp)
		featurelist.append(des.flatten())



	featurelist = np.asarray(featurelist)


	return featurelist
def getBriefFeatures(filelist):
	filelist.sort()
	featurelist = []
	#featurelist=np.array([])
	for i, imagepath in enumerate(filelist):
		print(" Status: %s / %s" %(i, len(filelist)), end="\r")
		img=cv2.imread(imagepath)
		
		# find and draw the keypoints
		width = 164
		height = 128
		dim = (width, height)

		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		# Initiate FAST detector
		star = cv2.xfeatures2d.StarDetector_create()
		# Initiate BRIEF extractor
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		# find the keypoints with STAR
		kp = star.detect(img,None)
		# compute the descriptors with BRIEF
		kp, des = brief.compute(img, kp)


	
		featurelist.append(des.flatten())


	featurelist = np.asarray(featurelist)
	return featurelist

def getOrbFeatures(filelist):
	filelist.sort()
	featurelist = []
	#featurelist=np.array([])
	for i, imagepath in enumerate(filelist):
		print(" Status: %s / %s" %(i, len(filelist)), end="\r")
		img=cv2.imread(imagepath,0)
		
		# find and draw the keypoints
		width = 164
		height = 128
		dim = (width, height)

		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		orb = cv2.ORB_create()
		# find the keypoints with ORB
		kp = orb.detect(img,None)
		# compute the descriptors with ORB
		kp, des = orb.compute(img, kp)


	
		featurelist.append(des.flatten())


	featurelist = np.asarray(featurelist)
	return featurelist

def showImage(image_path):
	img = cv2.imread(image_path)

if __name__ == '__main__':
	my_parser = argparse.ArgumentParser(description='List the content of a folder')

	my_parser.add_argument('image_path',metavar='path',type=str,help='the path to the image directory')
	my_parser.add_argument('image_feature_path',metavar='path',type=str,help='the path to save the image extracted features')
	my_parser.add_argument("save_filename",type=str,help="the save filename ")
	args = my_parser.parse_args()

	image_path=args.image_path
	image_feature_path=args.image_feature_path
	save_filename=args.save_filename

	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


	print("The paths are",image_path,image_feature_path)
	print(os.getcwd())
	foldername=os.path.join('/mnt/d/Himani-work/gsoc2020/dataset/ideology_extra200ms',image_path)
	print(foldername)
	files_path=loadFilePaths(foldername)
	print("length of files are ",len(files_path))

	#save_filename=os.path.join('/mnt/d/Himani-work/gsoc2020/code',save_filename)
	#showImage(files_path[0])
	
	"""
	gray_features=getGrayScaleFeatures(files_path)
	print("gray_features",gray_features,gray_features.shape)
	saveFeatures(gray_features,"_grayscale_",save_filename,image_feature_path)

	
	canny_features=getCannyFeatures(files_path)
	print("canny_features",canny_features,canny_features.shape)
	saveFeatures(canny_features,"_canny_",save_filename,image_feature_path)
	
	sobel_features=getSobelyFeatures(files_path)
	print("sobel_features",sobel_features,sobel_features.shape)
	saveFeatures(sobel_features,"_sobel_",save_filename,image_feature_path)

	prewitt_features=getPrewittyFeatures(files_path)
	print("prewitt_features",prewitt_features,prewitt_features.shape)
	saveFeatures(prewitt_features,"_prewitt_",save_filename,image_feature_path)

	hog_features=getHogFeatures(files_path)
	print("hog features",hog_features,hog_features.shape)
	saveFeatures(hog_features,"_hog_",save_filename,image_feature_path)

	fast_features=getFastFeatures(files_path)
	print("fast features",fast_features,fast_features.shape)
	saveFeatures(fast_features,"_fast_",save_filename,image_feature_path)
	
	brief_features=getBriefFeatures(files_path)
	print("brief_features",brief_features,brief_features.shape)
	saveFeatures(brief_features,"_brief_",save_filename,image_feature_path)
	"""
	orb_features=getOrbFeatures(files_path)
	print("orb_features",orb_features,orb_features.shape)
	saveFeatures(orb_features,"_orb_",save_filename,image_feature_path)
	

