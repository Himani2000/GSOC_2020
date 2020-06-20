#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import parselmouth
from numpy import unique
from numpy import where
from matplotlib import pyplot as plt
import librosa.display
from sklearn import metrics
from scipy.signal import lfilter, hamming
from textGrid_AudioTrim import readTextGridUpdate,audio_trimming
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def formantLpc(newWavFile_vowel):
    #preprocessing first step -->get the hamming window 
    x=newWavFile_vowel
    N = len(x)
    w = np.hamming(N)
    #preprocessing step 2 --> apply the window and the high pass filter 
    # Apply window and high pass filter.
    #issues to be fixed in this version of code is  
    #a) how to choose the order in the lpc 
    #b) how to filter out the resultant frequency array 
    # c) how many formants to get 
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    Fs =16000
    ncoeff = 2 + Fs / 1000
    print(ncoeff)
    A = librosa.core.lpc(x1, int(ncoeff))
    rts = np.roots(A)
    rts = rts[np.imag(rts) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * (Fs / (2 *  np.pi))
    frqs.sort()
    
    frqs=[i for i in frqs if(i>0.0 and i<=5500)]
    
    if(len(frqs)>=3):
        return frqs[0:3]
    else:
        return frqs
    



# In[4]:


def calculateFormantsPraat(formant,atTime):
    formants=[]
    for i in range(1,6):
      #  print(i)
        x=formant.get_value_at_time(i,atTime)
        if(x!=0 and np.isnan(x)==False):
            formants.append(x)
    print(formants)
    formants.sort()
    if(len(formants)>=3):
        return formants[0:3]
    else:
        return formants


# In[5]:


def getFormantsPraat(start_time,end_time,sound,formantType):
    point10=start_time+((end_time-start_time)*0.1)
    point20 = start_time + ((end_time-start_time)*0.2)
    point50 = start_time + ((end_time-start_time)*0.5)
    point80 = end_time - ((end_time-start_time)*0.2)
  
   # print("start and end time for formants",point20,point50,point80)
    formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
    if(formantType==50):
        formants=calculateFormantsPraat(formant,point50)
        if(np.isnan(formants).all()==True):
           
            point50=point50+point10
        formants=calculateFormantsPraat(formant,point50)
        
    elif(formantType==20):
        formants=calculateFormantsPraat(formant,point20)
        if(np.isnan(formants).all()==True):
            
            point20=point20+point10
        formants=calculateFormantsPraat(formant,point20)
        
    
    elif(formantType==80):
        formants=calculateFormantsPraat(formant,point80)
        if(np.isnan(formants).all()==True):
           
            point80=point80+point10
        formants=calculateFormantsPraat(formant,point80)
        
    
    
    
    return formants


# In[6]:


def extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,formantType,formantMethod):
    formants_vowel=[]
    filenames=[]
    SAMPLE_RATE=16000
    i=0

    for path in final_files:
        filenames.append(path)
        
        try:
            print(i)
            i=i+1
     
            textGrid_file_path=os.path.join(base_path_textGrid,path)+'.TextGrid'
            start_time_word,end_time_word,start_time_pronunciation_vowel,end_time_pronunciation_vowel=readTextGridUpdate(textGrid_file_path,word,pronunciation_vowel)
            
           # print(start_time_word,end_time_word,start_time_pronunciation_vowel,end_time_pronunciation_vowel)
            audio_file_path=os.path.join(base_path_audio,path)+'.wav'
            sound=parselmouth.Sound(audio_file_path)
            
            if(formantMethod=='praat'):
                sound=parselmouth.Sound(audio_file_path)
                formants=getFormantsPraat(start_time_pronunciation_vowel,end_time_pronunciation_vowel,sound,formantType)
            
            elif(formantMethod=='lpc'):
                newWavFile_vowel=audio_trimming(audio_file_path,start_time_pronunciation_vowel,end_time_pronunciation_vowel)
                formants=formantLpc(newWavFile_vowel)
            
            formants_vowel.append(formants)
           # print(formants_vowel)
           # print(filenames)
            
        except Exception as e:
            print(e)
    
    
    df_vowel=pd.DataFrame({'feature':formants_vowel})
    df_vowel=df_vowel.feature.apply(pd.Series)
    df_vowel['file_name']=filenames
    
    return df_vowel

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
    
    df=pd.read_excel("D://Himani-work/gsoc2020/dataset/spreadsheet_data/muslim_concordance_250_annotated.xls")
    df.dropna(inplace=True)
    all_correct_files=list(df['File Name'])
    
    print("total corrected files",len(all_correct_files))
    
    final_files=list(set(all_audio_files)&set(all_correct_files)&set(all_textGrid_files))
    print("final files",len(final_files))
    word="muslim"
    pronunciation_vowel="V"
    
         
    df_formant_20=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,20,'praat')
    df_formant_50=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,50,'praat')
    df_formant_80=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,80,'praat')
    df_formant_lpc=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,0,'lpc')
    
    
    df_formant_20.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_20_muslim_features_vowel.csv',index=False)

    df_formant_50.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_50_muslim_features_vowel.csv',index=False)

    df_formant_80.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_80_muslim_features_vowel.csv',index=False)

    df_formant_lpc.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_lpc_muslim_features_vowel.csv',index=False)
   
    
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

       
    df_formant_20=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,20,'praat')
    df_formant_50=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,50,'praat')
    df_formant_80=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,80,'praat')
    df_formant_lpc=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,0,'lpc')
    
    
    df_formant_20.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_20_ideology_five_features_vowel.csv',index=False)

    df_formant_50.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_50_ideology_five_features_vowel.csv',index=False)

    df_formant_80.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_80_ideology_five_features_vowel.csv',index=False)

    df_formant_lpc.to_csv('D://Himani-work/gsoc2020/dataset/Audio_features/formants_lpc_ideology_five_features_vowel.csv',index=False)
   
    if __name__=="__main__":
    # three variables are needed here the two  basepaths and the dataframe path
    # as an additional feature we can definitely ask for the feature to be extracted
    base_path_textGrid="D://Himani-work/gsoc2020/dataset/ideology_extra200ms/textgrids_for_extra_200ms_MAUS/"
    base_path_audio="D://Himani-work/gsoc2020/dataset/ideology_extra200ms/with_extra_200ms"
    
    textGridFiles=list(os.listdir(base_path_textGrid))
    audioFiles=list(os.listdir(base_path_audio))
    all_audio_files=list(pd.DataFrame({'audioFile':audioFiles})['audioFile'].str.split(".wav",expand=True).iloc[:,0])
    all_textGrid_files=list(pd.DataFrame({'textGridFile':textGridFiles})['textGridFile'].str.split(".TextGrid",expand=True).iloc[:,0])
    #print("total texGrid and audio files",len(all_audio_files),len(all_textGrid_files))
    
    df_ideology_200ms=pd.read_csv('D:/Himani-work/gsoc2020/dataset/spreadsheet_data/ideology_results_praat_formants_extracted_with_200ms_for_R.csv',sep='\t')
    all_correct_files=list(df_ideology_200ms['file'].str.split("\\",expand=True).iloc[:,-1].str.split(".wav",expand=True).iloc[:,0])
   # print("total corrected files",len(all_correct_files))
    
    final_files=list(set(all_audio_files)&set(all_correct_files)&set(all_textGrid_files))
    #print("final files",len(final_files))
    
     # Flow is like call the 
    word="ideology"
    pronunciation_vowel="aI"

    
   # df_formant_20=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,20,'praat')
   # df_formant_50=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,50,'praat')
   # df_formant_80=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,80,'praat')
   # df_formant_lpc=extractAllFormants(final_files,base_path_textGrid,base_path_audio,word,pronunciation_vowel,0,'lpc')
    
    
   

   

    

# In[ ]:





# In[ ]:




