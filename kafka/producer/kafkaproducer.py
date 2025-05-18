import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Load environment variables
bootstrap_servers = os.getenv("BOOTSTRAP_SERVERS", "kafka:9092")

# Function to connect to Kafka with retries
def connect_kafka():
    max_retries = 20
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            return producer
        except NoBrokersAvailable:
            print(f"Retry {attempt + 1}/{max_retries}: Kafka broker not available, retrying in 10 seconds...")
            time.sleep(10)
    raise Exception("Failed to connect to Kafka after retries")

# Connect to Kafka
producer = connect_kafka()

# Read CSV file
df_train = pd.read_csv('train20.csv')
# Function to send data to Kafka
def send_to_kafka(df, topic):
    for _, row in df.iterrows():
        data = row.to_dict()
        producer.send(topic, value=data)
        print(f"Sent to {topic}: {data}")
        time.sleep(1)  # Simulate delay

# Send train and test data
send_to_kafka(df_train, 'simulate_data_train')

# Close producer
producer.flush()
producer.close()
