from src.components.inference import ModelPrediction
from src.config.configuration import ConfigurationManager

video_path = r"D:\Ml Dl\Project\LipNet\artifacts\data_ingestion\data\s1\bbaf2n.mpg"
config=ConfigurationManager()
model_prediction_config=config.get_model_evaluation()

model=ModelPrediction(model_prediction_config)
pred_text = model.predict(video_path)

print("Predicted text : ",pred_text)

from fastapi import FastAPI, HTTPException, UploadFile, File

from fastapi.responses import JSONResponse
import shutil


app = FastAPI()

@app.get("/")
def root():
    return {"message": "This is the API of LipNet"}

@app.post("/predict")
def predict(video_file : UploadFile=File(...,description="It is the path of the video")):
    try:

        temp_path = f"temp_{video_file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        model=ModelPrediction(model_prediction_config)
        pred_text = model.predict(temp_path)

        return JSONResponse(status_code=200,content={"message":pred_text})
    
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))