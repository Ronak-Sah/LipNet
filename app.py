from src.components.inference import ModelPrediction
from src.config.configuration import ConfigurationManager

video_path = r""

config=ConfigurationManager()
model_prediction_config=config.get_model_evaluation()

model=ModelPrediction(model_prediction_config)
pred_text = model.predict(video_path)

print(pred_text)