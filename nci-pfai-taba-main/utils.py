import requests
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from io import StringIO
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from constants import *

def prepare_my_sql(sql_url, database_url):
    try:
        sql_engine = create_engine(sql_url)
        sql_engine.connect()
    except SQLAlchemyError as e:
        # Handle the SQLAlchemyError (or a more specific exception if needed)
        print(f"Error creating connection to MySQL: {e}")
        print("Check if the MySQL server is running and that the credentials and port are correct.")
        return False
        # Exit the program or take appropriate action
        # exit(1)
    print("Connection to MySQL server was successful.")

    # Check for the existence of the database and the table
    print("Checking for the existence of the database and the table...")
    engine = create_engine(database_url)
    # Check if the target database exists and create it if it doesn't
    try:
        engine.connect()
    except ProgrammingError:
        # Database doesn't exist, so let's create it
        create_database_query = text(f"CREATE DATABASE {target_database}")
        
        with sql_engine.connect() as connection:
            connection.execute(create_database_query)

    # Check if the tables exists and drop it if it does
    inspector = inspect(engine)
    if inspector.has_table(table_name_er):
        drop_table_query = text(f"DROP TABLE {table_name_er}")
        with engine.connect() as connection:
            connection.execute(drop_table_query)
    
    if inspector.has_table(table_name_ft):
        drop_table_query = text(f"DROP TABLE {table_name_ft}")
        with engine.connect() as connection:
            connection.execute(drop_table_query)
    
    if inspector.has_table(table_name_er_processed):
        drop_table_query = text(f"DROP TABLE {table_name_er_processed}")
        with engine.connect() as connection:
            connection.execute(drop_table_query)
    
    if inspector.has_table(table_name_ft_processed):
        drop_table_query = text(f"DROP TABLE {table_name_ft_processed}")
        with engine.connect() as connection:
            connection.execute(drop_table_query)

    print("Database and table are ready to be used.")
    return True

def extract_csv_data():
    """Download CSV data from a URL and return it as a string."""
    print("Downloading CSV data...")

    response_er = requests.get(csv_url_er)
    response_ft = requests.get(csv_url_ft)
    data_er = response_er.text
    data_ft = response_ft.text
    print("CSV data has been downloaded succesfully.")

    # Step 4: Parse CSV data using pandas
    print("Parsing CSV data...")
    er_data_types = {
        'Date' : 'Int64',
        'D1'   : 'object',
        'Value': 'Float64',
    }
    ft_data_types = {
        'Date' : 'object',
        'D0'   : 'object',
        'D1'   : 'object',
        'D2'   : 'object',
        'Value': 'Float64',
    }
    df_er = pd.read_csv(StringIO(data_er), na_values=['NA'], sep=';',  skiprows=3, dtype=er_data_types)
    df_ft = pd.read_csv(StringIO(data_ft), na_values=['NA'], sep=';',  skiprows=3, dtype=ft_data_types)
    return df_er, df_ft

def load_data_to_sql(df, database_url, table_name):
    # Batching and Inserting into MySQL database
    chunk_size = 1000
    engine = create_engine(database_url)

    # Iterate over chunks and insert into the database
    for i in range(0, len(df), chunk_size):
        df_chunk = df.iloc[i:i+chunk_size]
        df_chunk.to_sql(table_name, con=engine, if_exists='append', index=False)

def get_data_from_sql(database_url):
    # Query the data from the SQLite database
    engine = create_engine(database_url)
    query = f"SELECT * FROM {table_name_er}"
    df_from_db_er = pd.read_sql_query(query, engine)
    query = f"SELECT * FROM {table_name_ft}"
    df_from_db_ft = pd.read_sql_query(query, engine)

    # Display the retrieved data
    print("Exchange rates data retrieved from the SQL database:")
    print(df_from_db_er)

    print("Foreign trade data retrieved from the SQL database:")
    print(df_from_db_ft)

def process_data(database_url):
    # Query the data from the SQLite database
    query = f"SELECT * FROM {table_name_ft}"
    engine = create_engine(database_url)
    df_from_db = pd.read_sql_query(query, engine)

    # Display the retrieved data
    print("Data retrieved from the SQLite database:")
    print(df_from_db)
    df = df_from_db.rename(columns={'D0': 'Type', 'D1': 'Country or zone', 'D2': 'Value type', 'Value': 'Value'})
    type_dict = {
        'E': 'Import',
        'A': 'Export',
        'H': 'Trade surplus/deficit'
    }
    df['Type'] = df['Type'].map(type_dict)
    country_dict = {
        'EUROPE_T': 'Europe total',
        'EU_T': 'European union total',
        'EUROZONE_T': 'Euroarea total',
        'DE': 'Germany',
        'IT': 'Italy',
        'FR': 'France',
        'AT': 'Austria',
        'ES': 'Spain',
        'NL': 'Netherlands',
        'BE': 'Belgium',
        'IE': 'Ireland',
        'PL': 'Poland',
        'SE': 'Sweden',
        'CZ': 'Czech Republic',
        'GB': 'United Kingdom',
        'RU': 'Russia',
        'ASIA_T': 'Asia total',
        'ME_T': 'Middle East total',
        'AE': 'United Arab Emirates',
        'SAU': 'Saudi Arabia',
        'CN': 'China',
        'HK': 'Hong Kong',
        'JP': 'Japan',
        'SG': 'Singapore',
        'IN': 'India',
        'KR': 'South Korea',
        'TUR': 'Turkey',
        'NAM_T': 'North America total',
        'US': 'United States',
        'CA': 'Canada',
        'SAM_T': 'South America total',
        'BR': 'Brazil',
        'MX': 'Mexico',
        'AFRICA_T': 'Africa total',
        'ZA': 'South Africa',
        'EG': 'Egypt',
        'OCEANIA_T': 'Oceania total',
        'AU': 'Australia'
    }
    df['Country or zone'] = df['Country or zone'].map(country_dict)
    #Filter to only include value type WMF
    df = df[df['Value type'] == 'WMF']
    """
    value_type_dict = {
        'WMF': 'Value in CHF million',
        'VVP': 'Change in %',
    }
    df['Value type'] = df['Value type'].map(value_type_dict)
    """
    # Remove Value type column
    df = df.drop(columns=['Value type'])

    # Separate date into year and month (1997-1 -> 1997, 1)
    df['Year'] = df['Date'].str.split('-').str[0]
    df = df.drop(columns=['Date'])
    # Place Year column at the beginning of the dataframe and leave all the others the same
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    # Remove all rows with year minor than 2004
    df = df[df['Year'] >= '2004']
    # Sum by year, type and country (only summing the values)
    df = df.groupby(['Year', 'Type', 'Country or zone'])['Value'].sum().reset_index()
    # Load processed data to SQL in a new table called processed_data
    load_data_to_sql(df, database_url, table_name_ft_processed)

    # Display the retrieved data
    print("Processed data:")
    print(df)



    