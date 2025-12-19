from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import Data_ingestion
from src.logger import logger


class Data_Ingestion_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion()

        data_ingestion=Data_ingestion(data_ingestion_config)
        data_ingestion.download()
        data_ingestion.extract_zip_file()
        