# predict.py
import pickle
import numpy as np

class HousePriceModel:
    def __init__(self, model_path: str = "model.pkl"):
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✓ Model loaded successfully.")
        except FileNotFoundError:
            self.model = None
            print(f"✗ Error: {model_path} not found.")

    def predict(self, features) -> float:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        
        # Convert schema attributes into a 2D array structure for the model
        input_data = [[
            features.square_meters,
            features.bedrooms,
            features.bathrooms,
            features.year_built
        ]]
        
        prediction = self.model.predict(input_data)
        return float(prediction[0])

# Initialize a single instance to be imported across the app
model_runner = HousePriceModel()