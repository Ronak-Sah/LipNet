from src.entity import DataIngestionConfig
from src.entity import ModelTrainerConfig
from src.entity import ModelEvaluationConfig
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
    


    def get_model_trainer(self) -> ModelTrainerConfig:
        config = self.config.model_trainer                 # Extracts only the data_ingestion part of config.yaml.
        params = self.params.model_trainer
        create_directories([config.root_dir])               # Create data_ingestion.root directory

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            alignment_data_path=config.alignment_data_path,
            speaker_data_path=config.speaker_data_path,
            landmark_model_path=config.landmark_model_path,
            epochs= params.epochs,
            emb_dim= params.emb_dim,
            ffn_hidden= params.ffn_hidden,
            num_heads= params.num_heads,
            drop_prob= params.drop_prob,
            num_layers= params.num_layers,
            batch_size= params.batch_size,
            max_length= params.max_length
        )

        return model_trainer_config
    

    def get_model_evaluation(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation                 # Extracts only the data_ingestion part of config.yaml.
        params = self.params.model_evaluation
        create_directories([config.root_dir])               # Create data_ingestion.root directory

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            alignment_data_path=config.alignment_data_path,
            speaker_data_path=config.speaker_data_path,
            landmark_model_path=config.landmark_model_path,
            emb_dim= params.emb_dim,
            ffn_hidden= params.ffn_hidden,
            num_heads= params.num_heads,
            drop_prob= params.drop_prob,
            num_layers= params.num_layers,
            batch_size= params.batch_size,
            max_length= params.max_length
        )

        return model_evaluation_config