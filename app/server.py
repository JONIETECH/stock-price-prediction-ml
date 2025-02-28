from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from pydantic import BaseModel
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockFeatures(BaseModel):
    features: list[float]

# Load model and scaler
try:
    model = joblib.load('app/linear_regression_model.joblib')
    scaler = joblib.load('app/scaler.joblib')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

app = FastAPI()
# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/', response_class=HTMLResponse)
async def predict_form(
    request: Request,
    open_price: float = Form(...),
    high_price: float = Form(...),
    low_price: float = Form(...),
    volume: float = Form(...)
):
    try:
        # Prepare features
        features = np.array([[open_price, high_price, low_price, volume]])
        logger.info(f"Input features: {features}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        logger.info(f"Scaled features: {features_scaled}")
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        formatted_prediction = f"{prediction:.2f}"
        logger.info(f"Formatted prediction: {formatted_prediction}")
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": formatted_prediction
            }
        )
    except Exception as e:
        logger.error(f"Detailed error in predict_form: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"An error occurred while making the prediction: {str(e)}"
            }
        )

@app.post('/predict')
async def predict(data: StockFeatures):
    """
    Predicts the stock price based on given features.

        StockFeatures object containing:
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - volume: Trading volume

    Returns:
        dict: Predicted closing price
    """
    try:
        features = np.array(data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return {'predicted_price': float(prediction)}
    except Exception as e:
        logger.error(f"Error in predict API: {str(e)}")
        return {'error': f'An error occurred while making the prediction: {str(e)}'}
