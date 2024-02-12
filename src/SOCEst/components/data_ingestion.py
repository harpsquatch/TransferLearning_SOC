import os
import urllib.request as request
from zipfile import ZipFile
from SOCEst import logger
from SOCEst.utils.common import get_size
from pathlib import Path
from SOCEst.entity.config_entity import (DataIngestionConfig)    
from google_auth_oauthlib.flow import InstalledAppFlow
import gdown

#In this module we create a class which takes the configuration and contains all the nessary functions related to data ingestion in it. 
#This class takes just one argument, configuration and then all the operations are performed on the configuration paratmeters

class DataIngestion: 
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file_from_google_drive(self, file_id, dest_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, dest_path, quiet=False)
    
    def extract_zip(self, zip_path, unzip_dir):
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            
    def download_file(self):
        for source in self.config.source_URL:
            for name, file_id in source.items():
                local_data_file = f'{name}.zip'
                unzip_dir = f'artifacts\data_ingestion\{name}'
                if not os.path.exists(unzip_dir):  # Check if the directory already exists
                    if not os.path.exists(local_data_file):  # Check if the file already exists
                        self.download_file_from_google_drive(file_id, local_data_file)
                        self.extract_zip(local_data_file, unzip_dir)
                        os.remove(local_data_file)  # Remove the zip file after extraction
                    else:
                        logger.info(f"File {local_data_file} already exists.")
                else:
                    logger.info(f"Directory {unzip_dir} already exists.")

