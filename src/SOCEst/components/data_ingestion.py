import os
import urllib.request as request
import zipfile
from SOCEst import logger
from SOCEst.utils.common import get_size
from pathlib import Path
from SOCEst.entity.config_entity import (DataIngestionConfig)    



#In this module we create a class which takes the configuration and contains all the nessary functions related to data ingestion in it. 
#This class takes just one argument, configuration and then all the operations are performed on the configuration paratmeters

class DataIngestion: 
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    #In this class we define the method to download the data 
    def download_file(self):
        
            for source in self.config.source_URL:  #source_URL is the list of dictionaries, 
                for name, url in source.items():   #each dictionary contains the single key value and the URL of the data source
                    local_data_file = Path(self.config.root_dir) / f"{name}.zip"  # Correct path construction
                    
                    if not os.path.exists(local_data_file):
                        filename, headers = request.urlretrieve(
                            url = url,
                            filename = local_data_file
                        )
                        logger.info(f"{filename} download! with following info: \n{headers}")
                    else:
                        logger.info(f"File already exists of size: {get_size(Path(local_data_file))}")


    #In this class we define the method to extract the data

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        for source in self.config.source_URL:
            for name, url in source.items():
                local_data_file = Path(self.config.root_dir) / f"{name}.zip"  # Correct path construction
                unzip_path = self.config.unzip_dir
                os.makedirs(unzip_path, exist_ok=True)
                with zipfile.ZipFile(local_data_file, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)
  