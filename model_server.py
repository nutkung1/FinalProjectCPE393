import torch
import base64
import uvicorn
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pandas as pd
import os

app = FastAPI(title="EV Analysis API")

# Global variables for models
feature_extractor = None
classification_model = None
regression_model = None
ev_data = None


class ImageRequest(BaseModel):
    image_base64: str


class PredictionResponse(BaseModel):
    predicted_class: str
    top_predictions: dict
    electric_range_stats: dict = None


class RegressionRequest(BaseModel):
    model_year: int = Field(..., description="Vehicle model year")
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    base_msrp: float = Field(..., description="Base MSRP in USD")
    clean_alternative_fuel_vehicle_eligibility: str = Field(
        ..., description="Clean alternative fuel eligibility"
    )
    # Make vehicle_id optional with default value
    vehicle_id: str = Field(
        "unknown", description="Vehicle ID"
    )  # Changed from required to optional
    cafv_type: str = Field(..., description="CAFV type")
    electric_vehicle_type: str = Field(..., description="Type of electric vehicle")


class RegressionResponse(BaseModel):
    predicted_range: float
    confidence_interval: dict


@app.on_event("startup")
async def load_models():
    global feature_extractor, classification_model, regression_model, ev_data

    # Load EV CSV data
    try:
        csv_path = os.path.join(
            os.path.dirname(__file__), "data/Electric_Vehicle_Population_Data.csv"
        )
        global ev_data  # Explicitly declare ev_data as global again
        ev_data = pd.read_csv(csv_path)
        print(f"EV data loaded successfully with {len(ev_data)} records!")
    except Exception as e:
        print(f"Error loading EV data: {str(e)}")
        ev_data = None

    # Load classification model
    try:
        feature_extractor = AutoImageProcessor.from_pretrained(
            "dreamypancake/fine_tune_Car_ConvNeXTv2"
        )
        classification_model = AutoModelForImageClassification.from_pretrained(
            "dreamypancake/fine_tune_Car_ConvNeXTv2"
        )
        print("Classification model loaded successfully!")
    except Exception as e:
        print(f"Error loading classification model: {str(e)}")

    # Load regression model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "ml_model/best_model.pkl")
        with open(model_path, "rb") as f:
            regression_model = pickle.load(f)
        print("Regression model loaded successfully!")
    except Exception as e:
        print(f"Error loading regression model: {str(e)}")


MODEL_NAME_MAPPING = {
    "TESLA_X": "MODEL X",
    "TESLA_3": "MODEL 3",
    "TESLA_S": "MODEL S",
    "TESLA_Y": "MODEL Y",
    "CHEVROLET_BOLT": "BOLT EV",
}


@app.post("/predict", response_model=PredictionResponse)
async def predict_classification(request: ImageRequest):
    global feature_extractor, classification_model, ev_data

    if classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Process the image
        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = classification_model(**inputs)
            logits = outputs.logits

        # Get prediction
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = classification_model.config.id2label[predicted_class_idx]

        # Get probability scores
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get top predictions
        top_k = 3
        top_probs, top_indices = torch.topk(probs[0], top_k)

        results = {}
        for i in range(top_k):
            class_name = classification_model.config.id2label[top_indices[i].item()]
            probability = round(top_probs[i].item() * 100, 2)
            results[class_name] = probability

        # Parse the car make and model from the predicted class - ONLY ONCE
        parts = predicted_class.split("_", 1)
        make = parts[0] if len(parts) > 0 else ""
        model = MODEL_NAME_MAPPING.get(
            predicted_class, parts[1] if len(parts) > 1 else ""
        )

        # Print debugging information
        print(f"Predicted class: {predicted_class}")
        print(f"Parsed make: {make}")
        print(f"Parsed model: {model}")

        # Aggregate EV data if available
        range_stats = {}
        if ev_data is not None:
            # Print column names for debugging
            print(f"Available columns in data: {ev_data.columns.tolist()}")

            # Use correct column names (with proper capitalization)
            # Debug the data - assuming column is "Make" not "make"
            print(f"Available makes in data: {ev_data['Make'].unique()}")

            # Filter by make only for more reliable results
            filtered_data = ev_data[ev_data["Make"].str.upper() == make.upper()]

            print(f"Found {len(filtered_data)} vehicles matching make {make}")

            if len(filtered_data) > 0:
                avg_range = (
                    filtered_data["Electric Range"].mean()
                    if "Electric Range" in filtered_data.columns
                    else 0
                )
                max_range = (
                    filtered_data["Electric Range"].max()
                    if "Electric Range" in filtered_data.columns
                    else 0
                )
                min_range = (
                    filtered_data["Electric Range"].min()
                    if "Electric Range" in filtered_data.columns
                    else 0
                )
                count = len(filtered_data)

                range_stats = {
                    "average_range": float(avg_range) if not pd.isna(avg_range) else 0,
                    "max_range": float(max_range) if not pd.isna(max_range) else 0,
                    "min_range": float(min_range) if not pd.isna(min_range) else 0,
                    "vehicle_count": int(count),
                }

        return PredictionResponse(
            predicted_class=predicted_class,
            top_predictions=results,
            electric_range_stats=range_stats,
        )

    except Exception as e:
        import traceback

        print(traceback.format_exc())  # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict_range", response_model=RegressionResponse)
async def predict_regression(request: RegressionRequest):
    global regression_model

    if regression_model is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded")

    try:
        # Create input DataFrame from request
        input_data = {
            "model_year": [request.model_year],
            "make": [request.make],
            "model": [request.model],
            "base_msrp": [request.base_msrp],
            "clean_alternative_fuel_vehicle_eligibility": [
                request.clean_alternative_fuel_vehicle_eligibility
            ],
            "cafv_type": [request.cafv_type],
            "electric_vehicle_type": [request.electric_vehicle_type],
        }

        # Create DataFrame (without vehicle_id)
        input_df = pd.DataFrame(input_data)

        print(f"Raw input features: {input_df.to_dict('records')}")

        # Convert all column names to lowercase for consistency
        input_df.columns = input_df.columns.str.lower()

        # Identify categorical columns
        categorical_cols = [
            col for col in input_df.columns if input_df[col].dtype == "object"
        ]

        # Apply one-hot encoding
        input_encoded = pd.get_dummies(
            input_df, columns=categorical_cols, drop_first=True
        )

        # Get the feature names the model expects
        expected_features = regression_model.get_booster().feature_names
        print(f"Model expects {len(expected_features)} features")

        # Add missing columns with zeros
        for feature in expected_features:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0

        # Keep only columns the model knows about
        input_encoded = input_encoded[expected_features]

        print(f"Processed features shape: {input_encoded.shape}")

        # Make prediction
        prediction = regression_model.predict(input_encoded)[0]
        print(f"Predicted range: {prediction}")
        # Calculate confidence interval (10% range)
        lower_bound = max(0, prediction * 0.9)
        upper_bound = prediction * 1.1

        return RegressionResponse(
            predicted_range=float(prediction),
            confidence_interval={
                "lower": float(lower_bound),
                "upper": float(upper_bound),
            },
        )

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing regression request: {str(e)}"
        )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "classification_model_loaded": classification_model is not None,
        "regression_model_loaded": regression_model is not None,
    }


if __name__ == "__main__":
    uvicorn.run("model_server:app", host="0.0.0.0", port=8000, reload=False)
