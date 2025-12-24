from src.logger import logger 
from multiprocessing import freeze_support

from src.pipeline.stage_01_data_ingestion import Data_Ingestion_pipeline

# logger.info("Code Starts")

# Stage_Name="Data Ingestion Stage"

# try:
#     logger.info("=========================================================================================")
#     logger.info(f"                                  {Stage_Name} started ")
#     logger.info("=========================================================================================")
#     data_ingestion=Data_Ingestion_pipeline()
#     data_ingestion.main()
#     logger.info("=========================================================================================")
#     logger.info(f"                                  {Stage_Name} ended ")
#     logger.info("=========================================================================================")
# except Exception as e:
#     logger.exception(e)
#     raise e





from src.pipeline.stage_02_model_trainer import Model_Trainer_pipeline


# Stage_Name="Model training Stage"

# try:
#     logger.info("=========================================================================================")
#     logger.info(f"                                  {Stage_Name} started ")
#     logger.info("=========================================================================================")
#     model_trainer=Model_Trainer_pipeline()
#     model_trainer.main()
#     logger.info("=========================================================================================")
#     logger.info(f"                                  {Stage_Name} ended ")
#     logger.info("=========================================================================================")
# except Exception as e:
#     logger.exception(e)
#     raise e


from src.pipeline.stage_03_model_evaluation import Model_Evaluation_pipeline
def main():
    logger.info("Code Starts")
    # data_ingestion=Data_Ingestion_pipeline()
    # data_ingestion.main()
    # model_trainer = Model_Trainer_pipeline()
    # model_trainer.main()
    model_evaluation=Model_Evaluation_pipeline()
    model_evaluation.main()

if __name__ == "__main__":
    freeze_support()   # IMPORTANT for Windows
    main()
    



# Stage_Name="Model Evaluation Stage"

# try:
#     logger.info("=========================================================================================")
#     logger.info(f"                                  {Stage_Name} started ")
#     logger.info("=========================================================================================")
#     model_evaluation=Model_Evaluation_pipeline()
#     model_evaluation.main()
#     logger.info("=========================================================================================")
#     logger.info(f"                                  {Stage_Name} ended ")
#     logger.info("=========================================================================================")
# except Exception as e:
#     logger.exception(e)
#     raise e

