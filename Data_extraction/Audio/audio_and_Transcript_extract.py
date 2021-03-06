#importing all the headerfiles 
import pandas as pd
import numpy as np
import re
import requests
import urllib.parse

def update_StartEndTime(df,word_length=5):

    """
    function: The aim of the function is to extract new start ,end time and the transcripts of the audio. The new time is
    calculated by considering the words equal to (word_look) left and right to the query word.

    input: the original dataframe having all the details 

    output: a)start_timeList : containing the updated start time of all the audio files
            b)end_timeList : containg the updated end time of all the audio files
            c)transcripts: a list containing the updated transcript of all the audio files

    """
    start_timeList=[]
    end_timeList=[]
    transcripts=[]

  
    for i in range(len(df)):
        try:
            s=" "
            text1=" "
            text2=" "
            text3=df['Query item'][i]
            for j in range(3):
                
                if(df['Tagged context before'][i].split(' ')[-word_length+j:][0].split('/')[1:3][1]!='NA'):
                    s=".".join(df['Tagged context before'][i].split(' ')[-word_length+j:][0].split('/')[1:3])
                    text1=" ".join(df['Context before'][i].split(' ')[-word_length+j:])
                    
                    break

            start_timeList.append(s)
            e=" "
            for j in range(3):
               
                #print(df['Tagged context after'][i].split(' ')[word_length-j-1].split('/')[1:3],j,word_length-j-1)
                if(df['Tagged context after'][i].split(' ')[word_length-j-1].split('/')[3:][1]!='NA'):
                    e=".".join(df['Tagged context after'][i].split(' ')[word_length-j-1].split('/')[3:])
                    text2=" ".join(df['Context after'][i].split(' ')[0:word_length-j])
                    
                    break
            end_timeList.append(e)
            transcript=" ".join([text1,text3,text2])
            transcripts.append(transcript)          
            

            
        except Exception as e:
            start_timeList.append(" ")
            end_timeList.append(" ")
            transcripts.append(" ")
            print(e)

    return start_timeList,end_timeList,transcripts

def url_helper(start,end,url):
    #the function will update the url of the audio according to the new start 
    #and end time
    """
    function : This function is called by update_url to update the url. 

    input: It takes start time , end time and the original url 

    output : return  the updated url 

    working: It parse the url into parts. Made a dictionary of the part that will be changing in the url
    called query. Updated the parts of the query dictionary . Encoded the url.
    """
    url_parts = list(urllib.parse.urlparse(url))
    query = dict(urllib.parse.parse_qsl(url_parts[4]))
    query['start']=start
    query['end']=end
    url_parts[4] = urllib.parse.urlencode(query)
    url=urllib.parse.urlunparse(url_parts)
    return url

def update_url(df_update,df):
    """
    function: The aim of the fucntion is to update the audio url according to new updated start and end time.
    It calls the url_helper function to update the url.

    input: a) The new updated dataframe having the updated start and end times 
           b) The original dataframe having the urls that is to be updated

    output: a list of updated urls for all the audio files

    """
    urls_list=[]
    for i in range(len(df_update)):
        try:
            updated_urls=url_helper(df_update['start_time'][i],df_update['end_time'][i],df['Audio Snippet (long)'][i])
            urls_list.append(updated_urls)
        except Exception as e:
            print(e)
    return urls_list

def download_AudioDataset(file_path,df_update):
    """
    function: The functions takes the updated url and use requests library to download the new audio file.

    input: a)file path : a path where the user wants to save the new audio files 
           b)df update: updated dataframe having all the upated urls 

    output: No variable is returned .

    """

    for i in range(len(df_update)):
        print("audio ",i)
        url = df_update['wget_url'][i]
        r = requests.get(url, allow_redirects=True)
        open(file_path+df_update['File_Name'][i]+'.wav', 'wb').write(r.content)

def save_TranscriptDataset(file_path,df_update):
    """
    function: The function saves the transcripts to the txt files

    input: a)filepath: where the user wants to save the updated transcripts
           b)df_update : updated dataframe having all the updated transcripts


    """
    for i in range(len(df_update)):
        with open(file_path+df_update['File_Name'][i]+'.txt','w') as f:
            f.write(df_update['transcripts'][i])

if __name__=='__main__':
    #reading the excel file using pandas fucntion
    #later try to give this path as a command line argument

    #df=pd.read_excel("D:\Himani-work\gsoc2020\dataset\spreadsheet_data\ideology_concordance_with_rapidannotator_results_and_with_downloadlinks_manually_corrected_mit_timings_ergänzt.xlsx")
    
    df=pd.read_excel('/mnt/rds/redhen/gallina/projects/clustering/ideology_complete_Dataset.xlsx')
    # making a new dataframe for the new audio files 
    df_update=pd.DataFrame(columns=['start_time','end_time','transcripts','wget_url','base_url','id','no_of_hits','ai','ee','File_Name','Label'])
    
    
    #updating the other information from the dataset
    df_update['id']=list(df['ID VON HIER'])
    df_update['no_of_hits']=list(df['Number of hit'])
    df_update['ai']=list(df['ai'])
    df_update['ee']=list(df['ee'])
    df_update['base_url']=list(df['Audio Snippet (long)'])
    
    
    #calling the function
    start_timeList,end_timeList,transcripts=update_StartEndTime(df,5)
   # 
    #transcripts=update_Transcripts(df,5)

    print(len(transcripts),len(start_timeList),len(end_timeList))
    #updating the dataframe    
    df_update['start_time']=start_timeList
    df_update['end_time']=end_timeList
    df_update['transcripts']=transcripts
    
    df_update=df_update[df_update['start_time']!=" "]
    df_update=df_update[df_update['end_time']!=" "]
    
    df_update.reset_index(drop=True,inplace=True)
    
    urls_list=update_url(df_update)
    df_update['wget_url']=urls_list   
      
    #updating the labels here
    for i in range(len(df_update)):
        if(df_update['ai'][i]==1):
            label='ai'

        elif(df_update['ee'][i]==1):
            label='ee'

        elif(df_update['ai'][i]==0 and df_update['ee'][i]==0):
            label='DELETEME'
        
        df_update['Label'][i]=label
        
     #updating the file names here 
    df_update['File_Name']=[df_update['id'][i]+'_clip_'+str(int(df_update['no_of_hits'][i]))+'_'+df_update['Label'][i]+'__'+df_update['start_time'][i]+'-'+df_update['end_time'][i] for i in range(len(df_update))]

    #downloading the audio dataset here 
    file_path_audio='d:\\Himani-work\\gsoc2020\\code\\ideology_wav_5word_Dataset\\'
    download_AudioDataset(file_path_audio,df_update)

    #saving the transcripts
    #file_path_text='d:\\Himani-work\\gsoc2020\\code\\ideology_text_5word_Dataset\\'
    #save_TranscriptDataset(file_path_text,df_update)

    # at the end saving the new updated dataset i.e df_update
    df_update.to_csv('d:\\Himani-work\\gsoc2020\\code\\ideology_updated_5_Word_dataset.csv')
 


