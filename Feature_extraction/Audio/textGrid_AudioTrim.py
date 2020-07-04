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
    pattern_search = re.compile(word, re.I)
    #print(the_list)
    count=0
    try:
        text_grid= tgt.read_textgrid(filename)
    
    except Exception as FileNotFoundError:
        print("file does not exist")
        
    # calculating the start and end time for the word 
    # there can be two possibilitie a)if there is only one matched word b)if there are multiple matched words 
    mid_interval_len=5


    matched_word_intervals=text_grid.tiers[0].get_annotations_with_text(pattern=pattern_search, n=0, regex=True)
    print(matched_word_intervals)

    if(len(matched_word_intervals)==1):
        #case 1-> exactly one matched word thus simply get the start and end time 
        start_time_word=matched_word_intervals[0].start_time 
        end_time_word=matched_word_intervals[0].end_time
        #print(start_time_word,end_time_word)
    
    
    elif(len(matched_word_intervals)>=2):
    #case 2 -> more than two matched words 
    # now there can be two cases a) if the length is equal to 11 then we can have the 5th word as the ideology 
    # b) but if length is not equal to 11 (because of the several NAs that we removed )
        total_interval_len=len(text_grid.tiers[0].intervals)
        all_intervals=text_grid.tiers[0].intervals

        if(total_interval_len==11):
            start_time_word=all_intervals[5].start_time
            end_time_word=all_intervals[5].end_time

        elif(total_interval_len!=11):
            query_interval={}
            for i,sub_interval in enumerate(text_grid.tiers[0].intervals):
                if(sub_interval.text.lower()==word):

                    #print(sub_interval,i)
                    query_interval[i]=abs(mid_interval_len-i)

            interval_index=min(query_interval, key=query_interval.get)
            start_time_word=all_intervals[interval_index].start_time
            end_time_word=all_intervals[interval_index].end_time

   
    #calculating the start and the end time for the vowel 
    matched_vowel_intervals=text_grid.tiers[2].get_annotations_with_text(pattern=pronunciation_vowel, n=0, regex=False)
    print(matched_vowel_intervals)
    if(len(matched_vowel_intervals)==1):
       # print("length 1")
        start_time_pronunciation_vowel=matched_vowel_intervals[0].start_time
        end_time_pronunciation_vowel=matched_vowel_intervals[0].end_time
    elif(len(matched_vowel_intervals)>=2):
        for i in matched_vowel_intervals:
            start_time=i.start_time
            end_time=i.end_time

            if(i.start_time>=start_time_word and i.end_time<=end_time_word):

                start_time_pronunciation_vowel=start_time
                end_time_pronunciation_vowel=end_time
                break


    
    return start_time_word,end_time_word,start_time_pronunciation_vowel,end_time_pronunciation_vowel

    

def audio_trimming(audio_file,start_time,end_time):
    """
    function: The function aims to trim the audio file based on the start and the endtime 
    input: the input is the path of the audio file to be trimmed 
    output: the output is the numpy array of the trimmed audio 


    """
    y1,fs1=librosa.load(audio_file,sr=16000)
    #print("start and end time is ",start_time,end_time)
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
    return newWavFile

