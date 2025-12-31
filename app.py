from src.components.inference import ModelPrediction
from src.config.configuration import ConfigurationManager
from fastapi import FastAPI, HTTPException, UploadFile, File
import time
from fastapi.responses import JSONResponse
import shutil

config=ConfigurationManager()
model_prediction_config=config.get_model_evaluation()

model=ModelPrediction(model_prediction_config)

app = FastAPI()


@app.get("/")
def root():
    return {"message": "This is the API of LipNet"}

@app.post("/predict")
def predict(video_file : UploadFile=File(...,description="It is the path of the video")):
    start = time.perf_counter()

    try:

        temp_path = f"temp_{video_file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        model=ModelPrediction(model_prediction_config)
        pred_text = model.predict(temp_path)
        infer_time = (time.perf_counter() - start) * 1000
        return JSONResponse(status_code=200,content={
            "result":pred_text,
            "inference_ms": infer_time
        })
    
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))