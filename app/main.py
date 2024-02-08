import pandas as pd
from fastapi import FastAPI, Form, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.pipeline import PredictPipeline, CustomData

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
pipeline = PredictPipeline()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/manual_predict", response_class=HTMLResponse)
def render_manual_form(request: Request):
    return templates.TemplateResponse("manual_predict.html", {"request": request})

@app.post("/manual_predict", response_class=HTMLResponse)
def manual_predict(request: Request, type: str = Form(...),
                    air_temperature: float = Form(...),
                    process_temperature: float = Form(...),
                    rotational_speed: float = Form(...),
                    torque: float = Form(...),
                    tool_wear: float = Form(...)):
    custom_data = CustomData(type, air_temperature, process_temperature, rotational_speed, torque, tool_wear)
    data_df = custom_data.get_data_as_dataframe()
    prediction = pipeline.predict(data_df, manual=True)
    return templates.TemplateResponse("manual_predict.html", {"request": request, "predicted_class": prediction})

@app.get("/dataset_predict", response_class=HTMLResponse)
def render_dataset_form(request: Request):
    return templates.TemplateResponse("dataset_predict.html", {"request": request})

@app.post("/dataset_predict", response_class=HTMLResponse)
def predict_dataset(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        processed_df = pipeline.process_dataset(df)
        predictions = pipeline.predict(processed_df)
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "predicted_classes": predictions})
    except ValueError as e:
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "error_message": str(e)})