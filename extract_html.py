from bs4 import BeautifulSoup
import requests
import argparse


def extract_htmlFile(savefile):
    """
    This function is called from the shell script.

    function: the aim of the function is to extract the url from a tag present in the html file.
    It then uses requests library to extract the textGrid file from the url extracted.

    input: savefile: it is the name in which the user wants to save the file
    output: it doesnot return any variable


    """
    with open("demo.html", encoding="utf-8") as f:
        data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        soup.prettify()
        ul_list = soup.select("downloadlink")
        ul_list[0].get_text().strip()
        url_new= ul_list[0].get_text().strip()
        print(url_new)
    filepath='/mnt/d/Himani-work/gsoc2020/code/ideology_textGrid_5word_Dataset/'
   # filepath='D:\\Himani-work\\gsoc2020\\code\\ideology_textGrid_5word_Dataset\\'
    r = requests.get(url_new, allow_redirects=True)
    print("saving to the ",filepath+savefile+'.TextGrid')
    open(filepath+savefile+'.TextGrid', 'wb').write(r.content)

if __name__=='__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
    my_parser.add_argument('Path',metavar='path',type=str,help='the path to list')

# Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.Path
    print(input_path)
    extract_htmlFile(input_path)
