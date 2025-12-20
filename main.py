from src.logger import logger 

# from src.pipeline.stage_01_data_ingestion import Data_Ingestion_pipeline

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

logger.info("Code Starts")

Stage_Name="Model training Stage"

try:
    logger.info("=========================================================================================")
    logger.info(f"                                  {Stage_Name} started ")
    logger.info("=========================================================================================")
    model_trainer=Model_Trainer_pipeline()
    model_trainer.main()
    logger.info("=========================================================================================")
    logger.info(f"                                  {Stage_Name} ended ")
    logger.info("=========================================================================================")
except Exception as e:
    logger.exception(e)
    raise e