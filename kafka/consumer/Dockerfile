# Use a Python base image
FROM python:3.9-slim

# Install PostgreSQL dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the consumer code to the container
COPY kafkaconsumer.py /app/

# Command to run the consumer when the container starts
CMD ["python", "kafkaconsumer.py"]
