from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import Model_Evaluation
from src.logger import logger


class Model_Evaluation_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation()

        model_evaluation=Model_Evaluation(model_evaluation_config)
        model_evaluation.evaluate()