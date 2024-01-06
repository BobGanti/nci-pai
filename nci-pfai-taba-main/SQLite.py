import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from io import StringIO

# Step 1: Download CSV data
csv_url = "https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD"
response = requests.get(csv_url)
data = response.text

# Step 2: Parse CSV data using pandas
df = pd.read_csv(StringIO(data))

# Step 3: Get the path to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Step 4: Connect to the SQLite database using SQLAlchemy
database_url = f"sqlite:///{os.path.join(script_directory, 'taba_database.db')}"
engine = create_engine(database_url)

# Step 5: Store DataFrame in the SQLite database
df.to_sql("lacity", engine, index=False, if_exists="replace")

# Step 6: Print a message indicating success
print("Data has been successfully imported into the SQLite database.")

# Connect to the SQLite database using SQLAlchemy
# Query the data from the SQLite database
query = "SELECT * FROM lacity"
df_from_db = pd.read_sql_query(query, engine)

# Display the retrieved data
print("Data retrieved from the SQLite database:")
print(df_from_db)