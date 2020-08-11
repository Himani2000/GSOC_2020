#! /bin/bash 

#this script uses the curl command to fetch the textGrid from the webMAUS server 
#to fetch the textGrid file the curl command requires the .wav and the .txt file
#there are two different types of curl command  a)runMAUS basic b)G2PMAUS
#the aim is to pass the different audio and text file in each call
#the curl command requires the same filename for the audio and text file
#for this i first saved all the file name in a file_name2.txt file 
#traverse the file content and run the curl  command 
#after curl command python script extract_html is called

directory_path_audio=ideology_wav_5word_Dataset
directory_path_text=ideology_text_5word_Dataset

base_path_audio="$PWD/$directory_path_audio"
base_path_text="$PWD/$directory_path_text"

echo "the audio directory is $base_path_audio"
echo "the text directory is $base_path_text"

wav_ext=wav
text_ext=txt

file=file_name2.txt
for i in `cat $file`
do
  echo "$base_path_audio/$i.wav"
  echo "$base_path_text/$i.txt"
  filepath2="$base_path_audio/$i"
  filepath3="$base_path_text/$i"
  #if [ -f $filepath3 ]
  #  then 
   #     echo "File found $filepath3"
   # else 
   #     echo "not found $filepath3"
   # fi

 curl -v -X POST -H "content-type: multipart/form-data" -F SIGNAL=@$filepath2.wav -F LANGUAGE=eng-US -F TEXT=@$filepath3.txt "http://clarin.phonetik.uni-muenchen.de/BASWebServicesTest/services/runMAUSBasic" -o demo.html
 #curl -v -X POST -H "content-type: multipart/form-data" -F TEXT=@$filepath3.txt -F LANGUAGE=eng-US  -F SIGNAL=@$filepath2.wav  -F OUTFORMAT=TextGrid -F PIPE=G2P_CHUNKER_MAUS "http://clarin.phonetik.uni-muenchen.de/BASWebServicesTest/services/runPipeline" -o demo.html
 
 sleep 10s
 
 python3 extract_html.py $i
 done