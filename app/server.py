from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

model = joblib.load('app/linear_regression_model.joblib')

class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get('/', response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/', response_class=HTMLResponse)
async def predict_form(
    request: Request,
    feature1: float = Form(...),
    feature2: float = Form(...),
    feature3: float = Form(...),
    feature4: float = Form(...)
):
    features = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": class_name}
    )

# Keep the existing API endpoint for programmatic access
@app.post('/predict')
def predict(data: dict):
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"features": [1, 2, 3, 4]}

    Returns:
        dict: A dictionary containing the predicted class.
    """        
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}
