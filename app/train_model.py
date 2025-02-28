import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        df = pd.read_csv('app/stock_data.csv')
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Prepare features (X) and target (y)
        features = ['Open', 'High', 'Low', 'Volume']
        X = df[features]
        y = df['Close']
        
        # Split the data
        logger.info("Splitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        logger.info("Training the model...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        logger.info(f"Model R² score on training data: {train_score:.4f}")
        logger.info(f"Model R² score on test data: {test_score:.4f}")
        
        # Save both the model and scaler
        logger.info("Saving model and scaler...")
        joblib.dump(model, 'app/linear_regression_model.joblib')
        joblib.dump(scaler, 'app/scaler.joblib')
        
        logger.info("Model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 