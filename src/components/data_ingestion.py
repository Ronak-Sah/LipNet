import os
import urllib.request as request
from src.logger import logger
from src.utils.common import get_size
import zipfile
from src.entity import DataIngestionConfig
from pathlib import Path
import gdown

class Data_ingestion:
    def __init__(self,config: DataIngestionConfig):
        self.config= config


    def download(self):
        if not os.path.exists(self.config.local_data_file):

            gdown.download(
                self.config.source_url,
                self.config.local_data_file,
                quiet=False
            )
            
            logger.info(f"{self.config.local_data_file} download!")
        else :
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

   
