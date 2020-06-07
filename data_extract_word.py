import os
import pandas as pd
import requests
import urllib.parse


def get_audio_url(df):
    """
    function: This function update the base url according to the start ,end time and the name  of the file
    input: it requires the original dataframe having the data
    output:return the list of updated urls from the start , end and the file name 
    
    
    """
    base_url='http://pisa.vrnewsscape.ucla.edu/newsscape_wav_snippet.cgi?'
    url_parts=list(urllib.parse.urlparse(base_url))
    all_updated_urls=[]
    
    for i in range(len(df)):
        try:
            query=dict()                     
            query['file']=df['File Name Only'][i]
            query['start']=df['start_time'][i]
            query['end']=df['end_time'][i]
            url_parts[4]=urllib.parse.urlencode(query)
            url=urllib.parse.urlunparse(url_parts)
            all_updated_urls.append(url)
        except Exception as e:
            print(e)
            pass
    return all_updated_urls


def download_AudioDataset(file_path,df):
    
    """
    function: It will download the audio files from the update audio urls 
    input: a)the base file path b)the dataframe containing the updated url list
    output: It will download all the files in the destination filepath
    
    
    """

    for i in range(len(df)):
        print("audio ",i)
        url = df['audio_url'][i]
        r = requests.get(url, allow_redirects=True)
        open(file_path+df['File Name'][i]+'.wav', 'wb').write(r.content)


if __name__=='__main__':
    
    #reading the dataframe 
    
    df=pd.read_excel("D://Himani-work/gsoc2020/dataset/spreadsheet_data/muslim_concordance_250_random_annotated_in_rapid_annotator.xls")
    #dropping rows if no filename
    
    df.dropna(subset = ["File Name"], inplace=True)
   
    #seperating file name, start and end time 
    
    file_And_Time= df['File Name'].str.split("__", n = 1, expand = True) 
    df["File Name Only"]=file_And_Time[0]
    start_And_End=file_And_Time[1].str.split("-", n = 1, expand = True) 
    df["start_time"]=start_And_End[0]
    df["end_time"]=start_And_End[1]
    # calling the function to get the url 
    
    all_updated_urls=get_audio_url(df)
    df['audio_url']=all_updated_urls
    len(all_updated_urls)
    
    # calling the function to download all the files 
    file_path="D:\\Himani-work\\gsoc2020\\dataset\\muslim_dataset\\"
    download_AudioDataset(file_path,df)