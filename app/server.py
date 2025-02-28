from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime
import pytz

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
            stocks_data.append({
                'symbol': symbol,
                'price': f"{current_data.get('regularMarketPrice', 0):.2f}",
                'change': f"{current_data.get('regularMarketChangePercent', 0):.2f}",
                'last_updated': datetime.now(pytz.UTC).strftime("%H:%M:%S UTC")
            })
        except:
            continue
    
    return stocks_data

@app.get('/', response_class=HTMLResponse)
def read_root(request: Request):
    live_stocks = get_live_stocks()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "live_stocks": live_stocks
    })

@app.post('/', response_class=HTMLResponse)
async def predict_form(
    request: Request,
    open_price: float = Form(...),
    high_price: float = Form(...),
    low_price: float = Form(...),
    close_price: float = Form(...)
):
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

@app.post('/predict')
def predict(data: StockFeatures):
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
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {'predicted_price': float(prediction)}
