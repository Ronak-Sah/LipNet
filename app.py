from src.components.inference import ModelPrediction
from src.config.configuration import ConfigurationManager

video_path = r"D:\Ml Dl\Project\LipNet\artifacts\data_ingestion\data\s1\bbaf2n.mpg"
config=ConfigurationManager()
model_prediction_config=config.get_model_evaluation()

model=ModelPrediction(model_prediction_config)
pred_text = model.predict(video_path)

print("Predicted text : ",pred_text)