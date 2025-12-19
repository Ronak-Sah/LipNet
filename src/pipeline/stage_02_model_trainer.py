from src.config.configuration import ConfigurationManager
from src.components.model_trainer import Model_Trainer
from src.logger import logger


class Model_Trainer_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer()

        model_trainer=Model_Trainer(model_trainer_config)
        model_trainer.train()