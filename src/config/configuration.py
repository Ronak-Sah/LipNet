from src.entity import DataIngestionConfig


from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml,create_directories


class ConfigurationManager:
    def __init__(self,config_filepath= CONFIG_FILE_PATH,params_filepath= PARAMS_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    # Attribute for data ingestion
    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion                 # Extracts only the data_ingestion part of config.yaml.

        create_directories([config.root_dir])               # Create data_ingestion.root directory

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config