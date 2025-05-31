# 🚗⚡ Electric Vehicle Analysis MLOps Platform
A simple machine learning project that can identify electric cars from photos and predict their driving range. Built with modern MLOps practices for easy deployment and maintenance.

Slides: https://www.canva.com/design/DAGov1yV9Io/-zcjeiMGXPw5jMRL8jtWPQ/edit?utm_content=DAGov1yV9Io&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Project Front End: https://finalprojectcpe393.streamlit.app/

![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378432977603924099/image.png?ex=683c953f&is=683b43bf&hm=f274e36b7b835496dd6b3ce65931ccad664ac5104ad81d789cbd65779bcb358d&)

# 🌟 What This Project Does
📸 Car Recognition: Upload a photo of an electric car and get the make/model

📊 Range Prediction: Enter car details and get predicted driving range

🖥️ Easy Web Interface: Simple web app to use both features

🤖 Real-Time Data Ingsetion with Apache kafka and AWS RDS

🤖 Automated Everything: Automatic testing, building, and deployment

# 🏗️ How It Works (Simple Version)
What each part does:

Web Interface: Where users interact (like a website)

API Server: Handles requests and runs AI models

AI Models: One recognizes cars, one predicts range

Kafka: real-time data ingestion

Database: Stores car data

Automation: Keeps everything running smoothly

# 📁 Project Files (What's What)
📂 Main Files

├── app.py                    # The website users see

├── model_server.py           # The brain that runs AI models

├── requirements.txt          # List of needed software

├── Dockerfile.model          # Instructions to package the AI brain

├── Dockerfile.app           # Instructions to package the website

📂 AI Models

├── ml_model/                # Trained AI models stored here

└── notebooks/               # Where we experiment and clean data

📂 Automation

├── .github/workflows/       # Automatic testing and deployment

├── airflow/                 # Scheduled tasks (like model updates)

└── tests/                   # Code that checks if everything works

📂 Data

├── data/                    # Electric vehicle information

└── kafka/                   # Real-time data processing (advanced)

# 🚀 Getting Started (Easy Way)
What You Need

Docker (software that packages applications)

Git (for downloading code)

Step 1: Get the Code

Step 2: Start Everything
  Apache Kafka
  Airflow Docker-compose (Airflow + MLFlow)
  Ngrok + Backend

Step 3: Use the App

Open your browser to http://localhost:8501

Upload a car photo or enter car details

Get instant predictions!

🔧 What the AI Can Do

1. Recognize Cars from Photos
Send a photo, get back:

3. Predict Driving Range
Enter car details, get range prediction:

# 🧪 Quality Assurance
The project automatically:

✅ Tests all code before deployment

✅ Checks code quality and style

✅ Builds and packages everything correctly

✅ Reports any problems immediately

# 📊 How Good Are the AI Models?

## Car Recognition (ConvNeXTv2)

Accuracy: 95 %

Speed: Identifies cars in under 1 second

## Range Prediction (XGBoost)

Average Error: ±15 miles

Reliability: Works for 95% of electric vehicles

# 🔄 Automatic Operations (MLOps Magic)
What Happens When You Update Code:

Code Check: Automatically tests your changes

Quality Check: Makes sure code follows best practices

Build: Creates new versions of the app

Deploy: Updates the live application

Monitor: Watches for any problems

Scheduled Tasks:

Daily: Check for new car data

Weekly: Retrain models with the latest information

Monthly: Performance reports

🛠️ For Developers

Local Development

Adding New Features

Make changes to the code

Run tests: pytest

Push to GitHub

Automatic deployment handles the rest!

# 🎯 Why This Architecture?
Simple but Powerful:

Each part does one job well
Easy to fix problems
Can handle lots of users
Automatically stays up-to-date
Benefits for Users:

Fast predictions
Always available
Accurate results
Easy to use
Benefits for Developers:

Easy to maintain
Automatic testing
Quick deployments
Clear code organization

# 🚀 Production Features

High Availability: App stays running even if one part fails

Auto-Scaling: Handles more users automatically

Monitoring: Alerts if anything goes wrong

Security: Protects user data and prevents attacks

Backup: Regular data backups for safety
