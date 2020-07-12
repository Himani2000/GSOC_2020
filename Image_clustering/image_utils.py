import os
import numpy as np
import pandas as pd

def loadFilePaths(image_directory):
    
    files=os.listdir(image_directory)
    files_path=[os.path.join(image_directory,file) for file in files ]
    return files_path


def loadFeatures(filename):
    print("Loading file : ",filename)
    features= np.load(filename)
    return features