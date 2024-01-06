import requests
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from io import StringIO
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

# Step 1: Connect to the MySQL database using SQLAlchemy
print("Connecting to MySQL...")
# Configure the connection parameters as needed
user_name       = "root"
password        = ""
host            = "localhost"
port            = "3306"
sql_url         = f"mysql+mysqlconnector://{user_name}:{password}@{host}:{port}"
target_database = 'swiss_banking'
table_name      = "currency_exchange_rates"

# Check if there is a connection to the MySQL server
try:
    sql_engine = create_engine(sql_url)
    sql_engine.connect()
except SQLAlchemyError as e:
    # Handle the SQLAlchemyError (or a more specific exception if needed)
    print(f"Error creating connection to MySQL: {e}")
    print("Check if the MySQL server is running and that the credentials and port are correct.")
    # Exit the program or take appropriate action
    exit(1)
print("Connection to MySQL server was successful.")

# Step 2: Check for the existence of the database and the table
print("Checking for the existence of the database and the table...")
database_url = f"{sql_url}/{target_database}"
engine = create_engine(database_url)
# Check if the target database exists and create it if it doesn't
try:
    engine.connect()
except ProgrammingError:
    # Database doesn't exist, so let's create it
    create_database_query = text(f"CREATE DATABASE {target_database}")
    
    with sql_engine.connect() as connection:
        connection.execute(create_database_query)

# Check if the table exists and drop it if it does
inspector = inspect(engine)
if inspector.has_table(table_name):
    drop_table_query = text(f"DROP TABLE {table_name}")
    with engine.connect() as connection:
        connection.execute(drop_table_query)

print("Database and table are ready to be used.")

# Step 3: Download CSV data
print("Downloading CSV data...")
csv_url = "https://data.snb.ch/api/cube/devkua/data/csv/en"

response = requests.get(csv_url)
data = response.text
print("CSV data has been downloaded succesfully.")

# Step 4: Parse CSV data using pandas
print("Parsing CSV data...")
column_data_types = {
    'Date' : 'Int64',
    'D1'   : 'object',
    'Value': 'Float64',
}
df = pd.read_csv(StringIO(data), na_values=['NA'], sep=';',  skiprows=3, dtype=column_data_types)
df = df.dropna()

print("CSV data has been parsed succesfully.")

# Step 5: Store DataFrame in the MySQL database
print("Storing DataFrame in the MySQL database...")

# Determine the column types
sql_data_types = {
    col: 'DATETIME' if pd.api.types.is_datetime64_any_dtype(dtype) else ('VARCHAR(255)' if dtype == 'object' else dtype)
    for col, dtype in df.dtypes.items()
}
# Batching and Inserting into MySQL database
chunk_size = 1000

# Iterate over chunks and insert into the database
for i in range(0, len(df), chunk_size):
    df_chunk = df.iloc[i:i+chunk_size]
    df_chunk.to_sql(table_name, con=engine, if_exists='append', index=False)

print("Data has been successfully imported into the MySQL database.")

# Get column information for a table
"""
columns = inspector.get_columns(table_name)

for column in columns:
    print(column['name'], ' - ', column['type'])
"""

# Connect to the SQLite database using SQLAlchemy
# Query the data from the SQLite database
query = f"SELECT * FROM {table_name}"
df_from_db = pd.read_sql_query(query, engine)

# Display the retrieved data
print("Data retrieved from the SQLite database:")
print(df_from_db)