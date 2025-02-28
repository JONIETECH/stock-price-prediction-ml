from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime
import pytz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockFeatures(BaseModel):
    features: list[float]

model = joblib.load('app/linear_regression_model.joblib')

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

def get_live_stocks():
    """Get live stock data for popular stocks"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META']
    stocks_data = []
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            current_data = stock.info
            price = current_data.get('regularMarketPrice')
            change = current_data.get('regularMarketChangePercent')
            
            if price is not None and change is not None:
                stocks_data.append({
                    'symbol': symbol,
                    'price': f"{price:.2f}",
                    'change': f"{change:.2f}",
                    'last_updated': datetime.now(pytz.UTC).strftime("%H:%M:%S UTC")
                })
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    # Return at least an empty list if all API calls fail
    return stocks_data

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        live_stocks = get_live_stocks()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "live_stocks": live_stocks
        })
    except Exception as e:
        logger.error(f"Error in read_root: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "live_stocks": []
        })

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
        live_stocks = get_live_stocks()
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": formatted_prediction,
                "live_stocks": live_stocks
            }
        )
    except Exception as e:
        logger.error(f"Error in predict_form: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "An error occurred while making the prediction.",
                "live_stocks": []
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
