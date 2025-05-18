import os
import json
import psycopg2
from kafka import KafkaConsumer

# Load environment variables
rds_host = os.getenv("RDS_HOST")
rds_user = os.getenv("RDS_USER")
rds_password = os.getenv("RDS_PASSWORD")
rds_db = os.getenv("RDS_DB")
bootstrap_servers = os.getenv("BOOTSTRAP_SERVERS", "localhost:9092")

# Kafka Consumer
consumer = KafkaConsumer(
    'simulate_data_train',
    bootstrap_servers=bootstrap_servers,
    group_id='ev_consumer_group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    max_poll_records=100
)
# RDS Connection
conn = psycopg2.connect(
    host=rds_host,
    database=rds_db,
    user=rds_user,
    password=rds_password,
    port=5432
)
cursor = conn.cursor()

# Shared insert function for train
def insert_ev_data(data, table_name):
    try:
        query = f"""
        INSERT INTO {table_name} (
            vin, county, city, state, postal_code, model_year, make, model,
            ev_type, cafv_eligibility, electric_range, base_msrp, legislative_district,
            dol_vehicle_id, vehicle_location, electric_utility, census_tract
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (vin) DO NOTHING;
        """
        cursor.execute(query, tuple(data.values()))
        conn.commit()
        print(f"Inserted into {table_name}: {data}")
    except Exception as e:
        print(f"Error inserting into {table_name}: {e}")
        conn.rollback()
        
def create_table(table_name):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        vin NUMERIC PRIMARY KEY,
        county NUMERIC,
        city NUMERIC,
        state NUMERIC,
        postal_code NUMERIC,
        model_year NUMERIC,
        make NUMERIC,
        model NUMERIC,
        ev_type NUMERIC,
        cafv_eligibility NUMERIC,
        electric_range NUMERIC,
        base_msrp NUMERIC,
        legislative_district NUMERIC,
        dol_vehicle_id NUMERIC,
        vehicle_location NUMERIC,
        electric_utility NUMERIC,
        census_tract NUMERIC
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

# Consume messages and insert into the appropriate table
for msg in consumer:
    data = msg.value
    print(f"Received from topic {msg.topic}: {data}")
    if msg.topic == 'simulate_data_train':
        create_table('simulate_data_train')
        insert_ev_data(data, 'simulate_data_train')

cursor.close()
conn.close()
