# 🚗⚡ Electric Vehicle Analysis MLOps Platform
A simple machine learning project that can identify electric cars from photos and predict their driving range. Built with modern MLOps practices for easy deployment and maintenance.

Slides: https://www.canva.com/design/DAGov1yV9Io/-zcjeiMGXPw5jMRL8jtWPQ/edit?utm_content=DAGov1yV9Io&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Reports: https://docs.google.com/document/d/17qlmMI8Ot_4nT88Cjn9gaF8pYKDIK8VX0XQatznHP2U/edit?usp=sharing

Project Front End: https://finalprojectcpe393.streamlit.app/

# 🔍 Project Overview,
This MLOps platform combines computer vision and predictive modeling to analyze electric vehicles. Built with a modern tech stack including Streamlit, Apache Kafka, AWS RDS, and containerized with Docker, it demonstrates end-to-end machine learning operations from data ingestion to model deployment. The system features continuous integration/deployment through GitHub Actions and monitors performance with MLflow, making it both a practical tool and a showcase of MLOps best practices.

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

# Real-Time Data Ingestion
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378434866902994964/image.png?ex=683c9702&is=683b4582&hm=f84c78ce2c08d17926b7382de790294d72ee6ba746682a28528ecc9360266e73&)
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378434691752788111/image.png?ex=683c96d8&is=683b4558&hm=ddb35ebe016347c1449347843ae9d3d0774b90077b24c402cfa6343dfb8afb0a&)
# 🚀 Getting Started (Easy Way)
What You Need

Docker (software that packages applications)

Git (for downloading code)

Step 1: Clone the repository:
```bash
git clone https://github.com/nutkung1/FinalProjectCPE393.git
cd FinalProjectCPE393
```

Step 2: Start Everything
Apache Kafka
```bash
  cd kafka
  docker-compose start -f docker-compose.kafka.yaml up
```
Airflow Docker-compose (Airflow + MLFlow)
```bash
  docker-compose start -f docker-compose.airflow.yaml up
```
BackEnd
```bash
   python model_server.py
```
```bash
   ngrok 8000
```

Step 3: Use the App

Open your browser to http://localhost:8501 or https://finalprojectcpe393.streamlit.app/

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

# 🏗️ Apache Airflow and MLFlow
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378437006950137936/image.png?ex=683c9900&is=683b4780&hm=e660b7075d641600bf1fed42392c8f6961becda06733ba0c89841a720b1bf316&)
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378437073903554661/image.png?ex=683c9910&is=683b4790&hm=f68047625cc0cc40055f50260546d7ce3d47e3b25ec161baa564216441bbe45f&)
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378437126357647390/image.png?ex=683c991d&is=683b479d&hm=c55c33a7f24d5b79946c670fe667bf2d86e74e4362df5cfcae5221c6e6a727bd&)


# 📊 How Good Are the AI Models?

## Car Recognition (ConvNeXTv2)

Accuracy: 95 %

Speed: Identifies cars in under 1 second
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378433662588162150/image.png?ex=683c95e3&is=683b4463&hm=3836b21242a03fcc63de709c2f50f5b4dee95044fa206d1a39c64124553bb99a&)

## Range Prediction (XGBoost)

Average Error: ±15 miles

Reliability: Works for 95% of electric vehicles
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378433791277797456/image.png?ex=683c9601&is=683b4481&hm=cd0ad17a04b0626f775a9641d834a8b6e84c4b071f3c1b25fc5ea8c26500698a&)

## Residual MLP

MAE: 0.54

MSE: 5.19
![alt text](https://cdn.discordapp.com/attachments/1328192999238271027/1378433719630565406/image.png?ex=683c95f0&is=683b4470&hm=76f80c0fca4a085bcc8463057ae8d66f72e2c6c9b1eb6252917631eeca3e1c90&)

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
