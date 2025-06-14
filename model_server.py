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
    # Change this line to make it optional with a default value:
    vehicle_id: str = Field(default="unknown", description="Vehicle ID")
    cafv_type: str = Field(..., description="CAFV type")
    electric_vehicle_type: str = Field(..., description="Type of electric vehicle")


class RegressionResponse(BaseModel):
    predicted_range: float
    confidence_interval: dict


@app.on_event("startup")
async def load_models():
    global feature_extractor, classification_model, regression_model, ev_data

    feature_extractor = AutoImageProcessor.from_pretrained(
        "dreamypancake/fine_tune_Car_ConvNeXTv2"
    )
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
    if classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Use feature_extractor after image is defined
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


# Add these mappings at the global level, after your MODEL_NAME_MAPPING

# Mapping dictionaries for regression model
make_mapping = {
    "TESLA": 39,
    "NISSAN": 13,
    "CHEVROLET": 8,
    "BMW": 5,
    "FORD": 19,
    "TOYOTA": 41,
    "HONDA": 27,
    "VOLKSWAGEN": 44,
    "AUDI": 2,
    "KIA": 29,
    "HYUNDAI": 28,
    "VOLVO": 43,
    "MERCEDES-BENZ": 32,
    "PORSCHE": 35,
    "OTHER": 0,
}

model_mapping = {
    "MODEL 3": 98,
    "MODEL S": 99,
    "MODEL X": 100,
    "MODEL Y": 166,
    "LEAF": 101,
    "BOLT EV": 21,
}

cafv_mapping = {
    "Clean Alternative Fuel Vehicle Eligible": 1,
    "Not eligible": 0,
    "Unknown": 2,
}

ev_type_mapping = {
    "Battery Electric Vehicle": 0,
    "Battery Electric Vehicle (BEV)": 0,
    "Plug-in Hybrid Electric Vehicle": 1,
    "Plug-in Hybrid Electric Vehicle (PHEV)": 1,
}


@app.post("/predict_range", response_model=RegressionResponse)
async def predict_regression(request: RegressionRequest):
    if regression_model is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded")

    try:
        # Map text values to encoded values (keep your existing mappings)
        make_encoded = make_mapping.get(request.make.upper(), 0)
        model_encoded = model_mapping.get(request.model.upper(), 0)
        cafv_encoded = cafv_mapping.get(
            request.clean_alternative_fuel_vehicle_eligibility, 0
        )
        ev_type_encoded = ev_type_mapping.get(request.electric_vehicle_type, 0)

        print(f"Original make: {request.make} -> Encoded: {make_encoded}")
        print(f"Original model: {request.model} -> Encoded: {model_encoded}")

        # Get the EXACT feature names in the EXACT order from the model
        feature_names = regression_model.get_booster().feature_names
        print(f"Model expects these feature names: {feature_names}")

        # Create an empty DataFrame with the right columns in the right order
        input_df = pd.DataFrame(columns=feature_names)
        input_df.loc[0] = 0  # Initialize with zeros

        # Now assign values to the right columns
        input_df.at[0, "state"] = 0  # Default value
        input_df.at[0, "model_year"] = request.model_year
        input_df.at[0, "make"] = make_encoded
        input_df.at[0, "model"] = model_encoded
        input_df.at[0, "base_msrp"] = request.base_msrp
        input_df.at[0, "cafv_eligibility"] = cafv_encoded
        input_df.at[0, "ev_type"] = ev_type_encoded
        input_df.at[0, "census_tract"] = 53000000.0  # Default value

        print(f"Input features: {input_df.columns.tolist()}")

        prediction = regression_model.predict(input_df)[0]
        print(f"Prediction successful: {prediction}")

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
