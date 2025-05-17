import streamlit as st
from streamlit_option_menu import option_menu
import requests
import base64
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import json

# Set page config for consistent look
st.set_page_config(
    page_title="EV Analysis", layout="wide", initial_sidebar_state="collapsed"
)

# Custom CSS for app styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #1e88e5;
    }
    .metric-label {
        font-size: 14px;
        color: #757575;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "API_URL" not in st.session_state:
    st.session_state["API_URL"] = "http://localhost:8000"

if "API_KEY" not in st.session_state:
    st.session_state["API_KEY"] = ""

selected3 = option_menu(
    None,
    ["Classification", "Regression"],
    icons=["bi-diagram-3", "bi-graph-up"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0.5rem",
            "background": "linear-gradient(90deg, #2193b0, #6dd5ed)",
            "border-radius": "10px",
            "box-shadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
            "margin-bottom": "24px",
        },
        "icon": {"color": "white", "font-size": "20px", "margin-right": "8px"},
        "nav-link": {
            "font-size": "18px",
            "font-weight": "600",
            "text-align": "center",
            "padding": "12px 20px",
            "border-radius": "8px",
            "margin": "0 8px",
            "color": "rgba(255, 255, 255, 0.85)",
            "transition": "all 0.3s ease",
        },
        "nav-link-selected": {
            "background-color": "rgba(255, 255, 255, 0.2)",
            "color": "white",
            "font-weight": "700",
            "box-shadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
        },
    },
)

# API Configuration Section - Always visible
st.sidebar.title("API Configuration")
api_url = st.sidebar.text_input(
    "Ngrok URL",
    value=st.session_state["API_URL"],
    help="Enter the ngrok URL for your model API server",
    placeholder="https://your-ngrok-url.ngrok.io",
)

# Update session state if values changed
if api_url != st.session_state["API_URL"]:
    st.session_state["API_URL"] = api_url


# Prepare headers for authentication
def get_headers():
    headers = {}
    if st.session_state["API_KEY"]:
        headers["X-API-Key"] = st.session_state["API_KEY"]
    return headers


# Check API connection status
def check_api_connection():
    api_status = st.empty()
    try:
        response = requests.get(
            f"{st.session_state['API_URL']}/health", headers=get_headers(), timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("classification_model_loaded") and result.get(
                "regression_model_loaded"
            ):
                api_status.success("✓ Connected to model server (all models loaded)")
            elif result.get("classification_model_loaded"):
                api_status.warning(
                    "⚠️ Classification model loaded, but regression model may not be available"
                )
            elif result.get("regression_model_loaded"):
                api_status.warning(
                    "⚠️ Regression model loaded, but classification model may not be available"
                )
            else:
                api_status.warning(
                    "⚠️ Model server is reachable but models may not be loaded"
                )
        elif response.status_code == 401:
            api_status.error("❌ Authentication failed. Please check your API key.")
        else:
            api_status.warning(
                f"⚠️ API responded with status code: {response.status_code}"
            )

    except requests.exceptions.RequestException:
        api_status.error(
            "❌ Cannot connect to model server. Please check the Ngrok URL and ensure the server is running."
        )
    return api_status


if selected3 == "Classification":
    st.title("Car Image Classification")
    st.write("Upload an image of a car to identify its make and model.")
    stats = None
    # Check API connection
    api_status = check_api_connection()

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Image upload options
        uploaded_file = st.file_uploader(
            "Choose an image of a car...", type=["jpg", "jpeg", "png"]
        )

        image = None

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Car Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error opening image: {e}")

    with col2:
        if image is not None:
            with st.spinner("Analyzing car image..."):
                try:
                    # Convert image to base64 for API
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    # Send image to API
                    response = requests.post(
                        f"{st.session_state['API_URL']}/predict",
                        json={"image_base64": img_str},
                        headers=get_headers(),
                        timeout=30,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # Display results
                        st.subheader("Classification Results")
                        st.success(f"Predicted car: {result['predicted_class']}")

                        # Show top predictions
                        st.subheader("Top Predictions")
                        results = result["top_predictions"]
                        results_df = pd.DataFrame(
                            {
                                "Car Make/Model": list(results.keys()),
                                "Confidence (%)": list(results.values()),
                            }
                        )
                        st.table(results_df)

                        st.bar_chart(results)
                        if (
                            "electric_range_stats" in result
                            and result["electric_range_stats"]
                        ):
                            stats = result["electric_range_stats"]

                            # Add a horizontal bar chart to visualize the range distribution
                            range_data = pd.DataFrame(
                                {
                                    "Metric": ["Minimum", "Average", "Maximum"],
                                    "Range (miles)": [
                                        stats["min_range"],
                                        stats["average_range"],
                                        stats["max_range"],
                                    ],
                                }
                            )
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.error("Please check your connection to the model server.")
        else:
            st.info("Please upload a car image to classify.")
        # Show metric cards at the bottom - OUTSIDE columns, with proper condition check
    if stats is not None:
        # Create a row for metrics
        st.subheader("Electric Vehicle Metrics")
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-label">Average Range</p>
                    <p class="metric-value">{round(stats['average_range'])} mi</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_cols[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-label">Maximum Range</p>
                    <p class="metric-value">{round(stats['max_range'])} mi</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_cols[2]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-label">Minimum Range</p>
                    <p class="metric-value">{round(stats['min_range'])} mi</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_cols[3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-label">Vehicle Count</p>
                    <p class="metric-value">{stats['vehicle_count']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif selected3 == "Regression":
    st.title("Electric Vehicle Range Prediction")
    st.write("Enter vehicle details to predict its electric range in miles.")

    # Check API connection
    api_status = check_api_connection()

    # Create form for input
    with st.form("ev_data_form"):
        st.subheader("Vehicle Information")

        col1, col2 = st.columns(2)

        with col1:
            model_year = st.number_input(
                "Model Year", min_value=2010, max_value=2030, value=2023
            )
            make = st.selectbox(
                "Make",
                options=[
                    "Tesla",
                    "Nissan",
                    "Chevrolet",
                    "BMW",
                    "Ford",
                    "Toyota",
                    "Honda",
                    "Volkswagen",
                    "Audi",
                    "Other",
                ],
            )
            model = st.text_input("Model", value="Model 3")
            base_msrp = st.number_input(
                "Base MSRP ($)", min_value=10000, max_value=200000, value=45000
            )

        with col2:
            cafv_eligibility = st.selectbox(
                "Clean Alternative Fuel Vehicle Eligibility",
                options=[
                    "Clean Alternative Fuel Vehicle Eligible",
                    "Not eligible",
                    "Unknown",
                ],
            )
            vehicle_id = st.text_input("Vehicle ID", value="12345")
            cafv_type = st.selectbox(
                "CAFV Type",
                options=[
                    "Battery Electric Vehicle (BEV)",
                    "Plug-in Hybrid Electric Vehicle (PHEV)",
                    "Not Applicable",
                ],
            )
            ev_type = st.selectbox(
                "Electric Vehicle Type",
                options=[
                    "Battery Electric Vehicle (BEV)",
                    "Plug-in Hybrid Electric Vehicle (PHEV)",
                ],
            )

        submitted = st.form_submit_button("Predict Range")

    if submitted:
        with st.spinner("Predicting electric range..."):
            try:
                # Prepare data for API
                data = {
                    "model_year": model_year,
                    "make": make,
                    "model": model,
                    "base_msrp": base_msrp,
                    "clean_alternative_fuel_vehicle_eligibility": cafv_eligibility,
                    "vehicle_id": vehicle_id,
                    "cafv_type": cafv_type,
                    "electric_vehicle_type": ev_type,
                }

                # Send data to API
                response = requests.post(
                    f"{st.session_state['API_URL']}/predict_range",
                    json=data,
                    headers=get_headers(),
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display prediction results
                    st.subheader("Prediction Results")

                    # Create three columns for metrics
                    col1, col2, col3 = st.columns(3)

                    # Display predicted range with nice formatting
                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <p class="metric-label">Predicted Range</p>
                                <p class="metric-value">{round(result['predicted_range'])} mi</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Display confidence interval
                    with col2:
                        lower = round(result["confidence_interval"]["lower"])
                        upper = round(result["confidence_interval"]["upper"])
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <p class="metric-label">Range Estimate</p>
                                <p class="metric-value">{lower} - {upper} mi</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Display a charge estimate (80% of range)
                    with col3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <p class="metric-label">Est. Range at 80% Charge</p>
                                <p class="metric-value">{round(result['predicted_range'] * 0.8)} mi</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Visualize the range as a gauge chart
                    st.subheader("Range Visualization")

                    # Create range data for visualization
                    range_data = {
                        "category": ["Low Range", "Medium Range", "High Range"],
                        "values": [100, 250, max(400, result["predicted_range"] + 50)],
                    }

                    chart_data = pd.DataFrame(
                        {"Range (miles)": [result["predicted_range"]]}
                    )

                    st.bar_chart(chart_data)

                    # Add comparison with similar vehicles
                    st.subheader("How this compares")
                    comparison_data = {
                        "Vehicle Type": ["Compact EV", "This Vehicle", "Luxury EV"],
                        "Avg Range (miles)": [
                            150,
                            round(result["predicted_range"]),
                            300,
                        ],
                    }

                    comp_df = pd.DataFrame(comparison_data)
                    st.bar_chart(comp_df.set_index("Vehicle Type"))

                    # Add some context about the prediction
                    st.info(
                        f"""
                    The predicted range for your {model_year} {make} {model} is approximately {round(result['predicted_range'])} miles on a full charge.
                    
                    Factors that can affect actual range include:
                    - Driving conditions and terrain
                    - Weather and temperature
                    - Driving style and speed
                    - Vehicle load and passengers
                    - Battery age and condition
                    """
                    )

                elif response.status_code == 401:
                    st.error("Authentication failed. Please check your API key.")
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait before trying again.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Regression prediction error: {e}")
                st.error("Please check your connection to the model server.")

    # Dataset information
    with st.expander("About this model"):
        st.markdown(
            """
        ### Electric Vehicle Range Prediction Model
        
        This model predicts the electric range (in miles) of a vehicle based on its specifications. The model was trained using XGBoost on a dataset of electric vehicles registered in the United States.
        
        **Key features used for prediction:**
        - Model year
        - Vehicle make and model
        - Base MSRP (Manufacturer's Suggested Retail Price)
        - Vehicle type (BEV or PHEV)
        
        The model has been optimized for vehicles released between 2010-2023 and may be less accurate for future models with new battery technologies.
        """
        )
