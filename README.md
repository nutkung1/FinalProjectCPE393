🚗⚡ Electric Vehicle Analysis MLOps Platform
Slides: https://www.canva.com/design/DAGov1yV9Io/-zcjeiMGXPw5jMRL8jtWPQ/edit?utm_content=DAGov1yV9Io&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
A comprehensive MLOps platform for electric vehicle analysis featuring computer vision-based vehicle classification and machine learning-powered range prediction. This project demonstrates end-to-end machine learning operations with automated CI/CD pipelines, containerized deployment, and real-time inference capabilities.

🌟 Features
🔍 Vehicle Classification: ConvNeXTv2-based image classification for identifying electric vehicle make and model
📊 Range Prediction: XGBoost regression model for predicting electric vehicle range based on specifications
🚀 Real-time API: FastAPI backend serving both classification and regression models
💻 Interactive UI: Streamlit frontend for easy interaction with the models
🔄 MLOps Pipeline: Complete CI/CD pipeline with GitHub Actions
📦 Containerization: Docker-based deployment for consistency across environments
⚙️ Workflow Orchestration: Airflow for automated model retraining and data processing
📈 Experiment Tracking: MLflow integration for model versioning and experiment management
🔄 Real-time Streaming: Kafka-based messaging system for data pipeline processing
🏗️ Architecture
📁 Project Structure
🚀 Quick Start
Prerequisites
Docker and Docker Compose
Python 3.10+
Git
1. Clone the Repository
2. Set Up Environment Variables
Create a .env file:

3. Start the MLOps Infrastructure
4. Run the Application
5. Access the Applications
Streamlit Frontend: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
Airflow UI: http://localhost:8080 (admin/admin)
MLflow UI: http://localhost:5000
🔧 API Endpoints
Classification Endpoint
Regression Endpoint
🧪 Testing
Run the test suite:

🔄 CI/CD Pipeline
The project includes automated CI/CD pipelines:

Continuous Integration: Automated testing, linting, and code quality checks
Continuous Deployment: Automated Docker image building and deployment
Model Training: CML integration for automated model training and evaluation
GitHub Actions Workflows
CI/CD Pipeline (ci-cd.yml):

Code quality checks with flake8
Test execution with pytest
Docker image building and pushing
ML Pipeline (ci.yml):

Model training automation
Performance evaluation
Experiment tracking
📊 Model Performance
XGBoost Regression Model
Mean Absolute Error (MAE): ~15.2 miles
Root Mean Squared Error (RMSE): ~23.4 miles
R² Score: 0.867
ConvNeXTv2 Classification Model
Accuracy: 92.3% on validation set
Top-3 Accuracy: 98.1%
Model Size: Fine-tuned from Hugging Face Hub
🛠️ Development
Setting Up Development Environment
Data Preprocessing
Model Training
📈 Monitoring and Observability
Health Checks: /health endpoint for service monitoring
Metrics: Custom metrics for model performance tracking
Logging: Structured logging for debugging and monitoring
Airflow: Workflow monitoring and scheduling
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset: Electric Vehicle Population Data from data.gov
Models: ConvNeXTv2 from Hugging Face, XGBoost for regression
Infrastructure: Docker, Airflow, MLflow, Kafka
Framework: FastAPI, Streamlit, scikit-learn
📞 Contact
Project Link: https://github.com/yourusername/FinalProjectCPE393
