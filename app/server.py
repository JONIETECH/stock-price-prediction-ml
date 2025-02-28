from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockFeatures(BaseModel):
    features: list[float]

model = joblib.load('app/linear_regression_model.joblib')

app = FastAPI()
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
    close_price: float = Form(...)
):
    try:
        features = np.array([[open_price, high_price, low_price, close_price]])
        prediction = model.predict(features)[0]
        formatted_prediction = f"{prediction:.2f}"
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": formatted_prediction
            }
        )
    except Exception as e:
        logger.error(f"Error in predict_form: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "An error occurred while making the prediction."
            }
        )

@app.post('/predict')
async def predict(data: StockFeatures):
    """
    Predicts the stock price based on given features.

    Args:
        data: StockFeatures object containing:
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price

    Returns:
        dict: Predicted stock price
    """
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {'predicted_price': float(prediction)}
    except Exception as e:
        logger.error(f"Error in predict API: {str(e)}")
        return {'error': 'An error occurred while making the prediction'}
