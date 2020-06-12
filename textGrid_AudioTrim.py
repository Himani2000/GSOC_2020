import os
import numpy as np
import pandas as pd
import IPython.display as ipd
import tgt
import math
import sklearn
import librosa
import re
import scipy.io.wavfile
from numpy import unique
from numpy import where
from matplotlib import pyplot as plt
import librosa.display
from sklearn import metrics


def readTextGridUpdate(filename,word,pronunciation_vowel):
    """
    function: The function aims to find the specific interval having the word and the vowel
    input: the input is the path of the textGrid file to be read 
    output: the output is the start and end time of both the word and the vowel


    """
    the_list = re.compile(word, re.I)
    print(the_list)
    
    try:
        text_grid= tgt.read_textgrid(filename)
    
    except Exception as FileNotFoundError:
        print("file does not exist")
   
    for tier in text_grid.tiers:
        
        try:
            if(tier.get_annotations_with_text(pattern=the_list, n=0, regex=True)):
                interval1=tier.get_annotations_with_text(pattern=the_list, n=0, regex=True)
                start_time_word=interval1[0].start_time
                end_time_word=interval1[0].end_time
                
            

            elif(tier.get_annotations_with_text(pattern=pronunciation_vowel, n=0, regex=False)):
                interval2=tier.get_annotations_with_text(pattern=pronunciation_vowel, n=0, regex=False)
                if(len(interval2)==1):
                    start_time_pronunciation_vowel=interval2[0].start_time
                    end_time_pronunciation_vowel=interval2[0].end_time
                
                elif(len(interval2)>1):
                    for i in interval2:
                        start_time=i.start_time
                        end_time=i.end_time
                        if(i.start_time==start_time_word or abs(i.start_time-start_time_word)<=2):
                            start_time_pronunciation_vowel=start_time
                            end_time_pronunciation_vowel=end_time

        

        except Exception as e:
            print(e)
    return start_time_word,end_time_word,start_time_pronunciation_vowel,end_time_pronunciation_vowel

def audio_trimming(audio_file,start_time,end_time):
    """
    function: The function aims to trim the audio file based on the start and the endtime 
    input: the input is the path of the audio file to be trimmed 
    output: the output is the numpy array of the trimmed audio 


    """
    y1,fs1=librosa.load(audio_file,sr=16000)
   # fs1, y1 = scipy.io.wavfile.read(audio_file)
    #fs1,y1
    print("start and end time is ",start_time,end_time)
    l1=np.array([[start_time,end_time]])
    l1=np.ceil(l1*fs1)
    newWavFileAsList = []
    for elem in l1:
        startRead = elem[0]
        endRead = elem[1]
       
        if startRead >= y1.shape[0]:
            startRead = y1.shape[0]-1
        if endRead >= y1.shape[0]:
            endRead = y1.shape[0]-1
        newWavFileAsList.extend(y1[int(startRead):int(endRead)])

    newWavFile = np.array(newWavFileAsList)
    print(newWavFile.shape)
   # scipy.io.wavfile.write('test5.wav', fs1, newWavFile)
    #print("file saved ")
    return newWavFile

def toFeature(file_paths,base_path_textGrid,base_path_audio,word,pronunciation_vowel):
    """
    function: The function loops through all the files and extract the MFCC feature for both the word and the vowel
    
    input: The input is all the final files(see the main function for the explanation) name along with the base path
    of the audio and the textGrid file and also the word and vowel 
    
    output: The dataframe of mfcc features for both the word and the vowel are returned.


    """
    
    features_word=[]
    features_vowel=[]
    filenames=[]
    SAMPLE_RATE=16000
    i=0

    for path in file_paths:
        filenames.append(path)
        
        try:
            print(i)
            i=i+1
        
            #X, sample_rate = librosa.load(os.path.join(base_path,path),sr=SAMPLE_RATE)
            
            textGrid_file_path=os.path.join(base_path_textGrid,path)+'.TextGrid'
            start_time_word,end_time_word,start_time_pronunciation_vowel,end_time_pronunciation_vowel=readTextGridUpdate(textGrid_file_path,word,pronunciation_vowel)
            
            print(start_time_word,end_time_word,start_time_pronunciation_vowel,end_time_pronunciation_vowel)
            audio_file_path=os.path.join(base_path_audio,path)+'.wav'
            
            newWavFile_word=audio_trimming(audio_file_path,start_time_word,end_time_word)
            newWavFile_vowel=audio_trimming(audio_file_path,start_time_pronunciation_vowel,end_time_pronunciation_vowel)
            
            sample_rate = np.array(SAMPLE_RATE)
            mfccs_word = np.mean(librosa.feature.mfcc(y=newWavFile_word,sr=sample_rate,n_mfcc=13),axis=1)
            mfccs_vowel = np.mean(librosa.feature.mfcc(y=newWavFile_vowel,sr=sample_rate,n_mfcc=13),axis=1)
            
            features_word.append(mfccs_word)
            features_vowel.append(mfccs_vowel)
            
        except:
            print("Exception is caused because of the file ",path)
            features_word.append()
            features_vowel.append()
            
    df_features_word,df_features_vowel=make_dataframe(filenames,features_word,features_vowel)
    return df_features_word,df_features_vowel

def make_dataframe(filenames,features_word,features_vowel):
    """
    function: The aim is to contruct the  mfcc features dataframe for both word and the vowel
    input: the input is the features list of word and vowel
    output: the output is the feature dataframe of both word and the vowel


    """


    df_word=pd.DataFrame({'feature':features_word})
    df_vowel=pd.DataFrame({'feature':features_vowel})
    
    
    
    df_word=df_word.feature.apply(pd.Series)
    df_vowel=df_vowel.feature.apply(pd.Series)
    
    
    df_word['file_name']=filenames
    df_vowel['file_name']=filenames
    
    return df_word,df_vowel
    
if __name__=="__main__":
   
    
    # three variables are needed here the two  basepaths and the dataframe path
    # as an additional feature we can definitely ask for the feature to be extracted
    base_path_textGrid="D://Himani-work/gsoc2020/dataset/ideology_five_words_version2/ideology_textGrid_five_dataset_version2/"
    base_path_audio="D://Himani-work/gsoc2020/dataset/ideology_five_words_version2/ideology_wav_five_dataset_version2/"
    
    
   
    
    textGridFiles=list(os.listdir(base_path_textGrid))
    audioFiles=list(os.listdir(base_path_audio))
    all_audio_files=list(pd.DataFrame({'audioFile':audioFiles})['audioFile'].str.split(".wav",expand=True).iloc[:,0])
    all_textGrid_files=list(pd.DataFrame({'textGridFile':textGridFiles})['textGridFile'].str.split(".TextGrid",expand=True).iloc[:,0])
    print("total texGrid and audio files",len(all_audio_files),len(all_textGrid_files))
    
    
    
    final_files=list(set(all_audio_files)&set(all_textGrid_files))
    print("final files",len(final_files))
    
    # Flow is like call the 
    word=r"ideology"
    pronunciation_vowel="aI"


    df_features_word,df_features_vowel=toFeature(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel)

    df_features_word.to_csv('mfcc0_ideology_five_features_word.csv',index=False)
    df_features_vowel.to_csv('mfcc0_ideology_five_features_vowel.csv',index=False)
"""
#this is for muslim dataset
if __name__=="__main__":
   
    # three variables are needed here the two  basepaths and the dataframe path
    # as an additional feature we can definitely ask for the feature to be extracted
    base_path_textGrid="D://Himani-work/gsoc2020/dataset/muslim_dataset_version1/muslim_textGrid_dataset/"
    base_path_audio="D://Himani-work/gsoc2020/dataset/muslim_dataset_version1/muslim_wav_dataset/"
    
    textGridFiles=list(os.listdir(base_path_textGrid))
    audioFiles=list(os.listdir(base_path_audio))
    all_audio_files=list(pd.DataFrame({'audioFile':audioFiles})['audioFile'].str.split(".wav",expand=True).iloc[:,0])
    all_textGrid_files=list(pd.DataFrame({'textGridFile':textGridFiles})['textGridFile'].str.split(".TextGrid",expand=True).iloc[:,0])
    print("total texGrid and audio files",len(all_audio_files),len(all_textGrid_files))
    
    df=pd.read_excel("D://Himani-work/gsoc2020/dataset/spreadsheet_data/muslim_concordance_250_random_annotated_in_rapid_annotator.xls")
    df.dropna(inplace=True)
    all_correct_files=list(df['File Name'])
    
    print("total corrected files",len(all_correct_files))
    
    final_files=list(set(all_audio_files)&set(all_correct_files)&set(all_textGrid_files))
    print("final files",len(final_files))
    word="muslim"
    pronunciation_vowel="V"
    df_features_word,df_features_vowel=toFeature(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel)

    df_features_word.to_csv('mfcc0_muslim_features_word.csv',index=False)
    df_features_vowel.to_csv('mfcc0_muslim_features_vowel.csv',index=False)
    
    
"""