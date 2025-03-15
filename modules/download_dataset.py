import os
from zipfile import ZipFile

def download_dataset():
    os.system('kaggle datasets download kazanova/sentiment140')
    file_name = "sentiment140.zip"
    with ZipFile(file_name, 'r') as zip:
        zip.extractall()
        print('The data is extracted')

# Call the function to download and extract the dataset
if __name__ == "__main__":
    download_dataset()