import inline as inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
from mongo_model import MongoModel
from helper_methods import *
from data_layer import MongoDB
from io import StringIO

class GraphModel:
    def __init__(self):
        pass

    def david_raw_file(self):

        dl_url = "https://data.snb.ch/api/cube/auvercurra/data/csv/en"
        coll_name = "david_coll"
        def store_david_data():
            if input("Do you want to download the dataset (y/n)? ").lower() == "y":
                print("Downloading and Storing Raw Data...")
                snb_david_csv = Helper.download_file(dl_url)
                data_io = Helper.compress_if_necessary(snb_david_csv)  # returns bytes data compressed or not
                db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name=coll_name)
                return db.savefile_in_db(data_io)  # returns inserted_id

        def load_saved_david_data():
            inserted_id = "6590d5a4655571e0a80959e5"
            db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name=coll_name)
            d = db.loadfile_from_db()
            # Checking if the retrieved data was compressed before storing. If so, decompresses
            if Helper.is_zlib_compressed(d):
                d = zlib.decompress(d)
            d = d.decode()  # string data was encoded to bytes before storing. Decode it back to string
            d_io = StringIO(d)  # Convert the string data to a file-like obj for pd to be able to read

            return pd.read_csv(d_io, delimiter=';', skiprows=2)  # Skip 1st two rows (They're non-informative)

    def store_david_data(self):
        csv_url = "https://data.snb.ch/api/cube/auvercurra/data/csv/en"
        coll_name = "david_coll"
        if input("Do you want to download the dataset (y/n)? ").lower() == "y":
            print("Downloading and Storing Raw Data...")
            snb_david_csv = Helper.download_file(csv_url)
            data_io = Helper.compress_if_necessary(snb_david_csv)  # returns bytes data compressed or not
            db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name=coll_name)
            inserted_id = db.savefile_in_db(data_io,)  # returns inserted_id

    def load_saved_david_data(self):
        coll_name = "david_coll"
        inserted_id = "6590d5a4655571e0a80959e5"
        db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name=coll_name)
        d = db.loadfile_from_db()
        # Checking if the retrieved data was compressed before storing. If so, decompresses
        if Helper.is_zlib_compressed(d):
            d = zlib.decompress(d)
        d = d.decode()  # string data was encoded to bytes before storing. Decode it back to string
        d_io = StringIO(d)  # Convert the string data to a file-like obj for pd to be able to read

        df = pd.read_csv(d_io, delimiter=';', skiprows=2)  # Skip 1st two rows (They're non-informative)




# Neo4j database connection details
# Please change this details to your corresponding server information
db_url = 'bolt://localhost:7687'
db_username = 'neo4j'
db_password = '12345678'


connection_string = "mongodb://localhost:27017/"
db_name = "mongoGraphDb"
coll_name = "graph_collection"
url = "https://data.snb.ch/api/cube/auvercurra/data/csv/en"

# Establishing a connection with Neo4j database
# NB: Ensure your Neo4j database server is running before running the cell
try:
    # py2neo
    graph = Graph(db_url, auth=(db_username, db_password))
    # neo4j
    driver = GraphDatabase.driver(uri=db_url, auth=(db_username, db_password))
    session = driver.session()
except:
    print('Please ensure your graph database server is running and accurate information are provided')


def import_csv():
    if input("Do you want to download the dataset (y/n)? ").lower() == "y":  # "y" if data had already been downloaded
        print("Downloading and Storing Raw Data...")
        snb_graph_csv = Helper.download_file(url)
        data_io = Helper.compress_if_necessary(snb_graph_csv)  # returns bytes data compressed or not
        db = MongoDB(db_name=db_name, cnn_str=connection_string, collection_name=coll_name)
        return db.savefile_in_db(data_io)  # returns inserted_id


def load_saved_raw_data():
    db = MongoDB(db_name=db_name, cnn_str=connection_string, collection_name=coll_name)
    d = db.loadfile_from_db()
    if Helper.is_zlib_compressed(d): # Checking if the retrieved data was compressed before storing. If so, decompresses
        d = zlib.decompress(d)
    d = d.decode()  # string data was encoded to bytes before storing. Decode it back to string
    d_io = StringIO(d)  # Convert the string data to a file-like obj for pd to be able to read

    return pd.read_csv(d_io, delimiter=';', skiprows=2)  # Returns df, Skips 1st two rows (They're non-informative)


# Printing the dataset shape
def data_shape(df):
    print(f'Shape of data: {df.shape}\n')


# Printing the dataset information
def data_details(df):
    print(f'{df.describe()}\n{df.info()}\n')


# Checking for missing values
def check_na(df):
    print(f'NULL COUNT\n{df.isna().sum()}\n')


# Viewing samples of the dataset
def data_preview(df, num):
    print(f'{df.head(num)}\n')


# Getting the latest year with missing value
def max_empty_entry_year(df):
    empty_entry_years = []

    # Iterate over the data
    for i in range(len(df)):

        # Test if the value is empty
        if pd.isna(df['Value'][i]) == True:
            temp_year = df['Date'][i]
            empty_entry_years.append(temp_year)
    return max(empty_entry_years)

# Acquire sequential years with complete values
def get_complete_year(df):
    # Get the latest year with missing value
    start_year = max_empty_entry_year(df)

    # Get the last year on the data
    end_year = df.iloc[0, -1]

    # Extract years with complete data between the latest year with missing value and last year on the data
    new_df = df[((df['Date'] > start_year) & (df['Date'] <= end_year))]
    new_df.reset_index(drop=True, inplace=True)
    return new_df

# Rename columns
def col_rename(df, dic):
    df.rename(columns=dic, inplace=True)


# Replace values in the dataframe
def replace_values(df, dic):
    for temp_col in dic:
        df[temp_col] = df[temp_col].map(dic[temp_col]).fillna(df[temp_col])


# Extract unique values in the dataframe
def extract_unique_values(df, col):
    for i in col:
        if i == 'Category':
            category = df[i].unique()
        elif i == 'Currency':
            currency = df[i].unique()
        elif i == 'Investment':
            investment = df[i].unique()
    return category, currency, investment


# Plot trend line
def plot_trend(df, layer, layer1, layer2):
    # Duplicate the dataframe
    temp_df = df.copy(deep=True)

    # Scaling of 'Value' column
    temp_df['Value'] = temp_df['Value'] / 1000

    # Filter the data
    for i in layer1:
        filter_df_1 = temp_df[temp_df[layer[0]] == i]
        for j in layer2:
            filter_df_2 = filter_df_1[filter_df_1[layer[1]] == j]

            # Check if the dataframe is empty
            if filter_df_2.empty == True:
                print(f'There is no {i} for {j}.')
                continue

            # Set the plot style (optional)
            sns.set(style="whitegrid")

            # Plot the trend using Seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size

            for _, val in enumerate(filter_df_2[layer[2]].unique()):
                filter_df_3 = filter_df_2[filter_df_2[layer[2]] == val]
                sns.lineplot(x='Date', y='Value', data=filter_df_3, label=val)

            # Customize the plot
            plt.xticks(temp_df['Date'].unique(), rotation=45)
            plt.title(f'{j} ({i})')
            plt.xlabel('Year')
            plt.ylabel('Values (in Billions)')
            plt.legend()

            # Show the plot
            plt.show()


# Plot stacked barchart
def plot_barchart(df, layer, layer1, layer2):
    # Duplicate the dataframe
    temp_df = df.copy(deep=True)

    # Scaling of 'Value' column
    temp_df['Value'] = temp_df['Value'] / 1000

    # Filter the data
    for i in layer1:
        filter_df_1 = temp_df[temp_df[layer[0]] == i]
        for j in layer2:
            filter_df_2 = filter_df_1[filter_df_1[layer[1]] == j]

            # Check if the dataframe is empty
            if filter_df_2.empty == True:
                print(f'There is no {i} for {j}.')
                continue

            # Set the plot style (optional)
            sns.set(style="whitegrid")

            # Plot the trend using Seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size

            for _, val in enumerate(filter_df_2[layer[2]].unique()):
                filter_df_3 = filter_df_2[filter_df_2[layer[2]] == val]

                # Create the stacked bar chart
                plt.bar(filter_df_3['Date'], filter_df_3['Value'])

            # Customize the plot
            plt.xticks(temp_df['Date'].unique(), rotation=45)
            plt.title(f'{i} in {j}')
            plt.xticks(rotation=45)
            plt.xlabel('Year')
            plt.ylabel('Values (in Billions)')
            plt.legend(filter_df_2[layer[2]].unique())

            # Show the plot
            plt.show()


# Get minimum value with the corresponding year
def get_min_value(df):
    # Get index of minimum value
    idx = df['Value'].idxmin()

    # Get minimum value
    value = min(df['Value'])

    # Get minimum year
    year = df['Date'][idx]
    return value, year


# Get maximum value with the corresponding year
def get_max_value(df):
    # Get index of maximum value
    idx = df['Value'].idxmax()

    # Get maximum value
    value = max(df['Value'])

    # Get maximum year
    year = df['Date'][idx]
    return value, year


# Get mean value
def get_mean_value(df):
    mean = df['Value'].mean()
    return mean

# Get dataset statics information
def statics_arr(df, col):
    tab = []

    # Get unique values in different columns
    category, currency, investment = extract_unique_values(df, col)

    for i in category:
        for j in investment:
            for k in currency:

                # Filtering the database
                temp_df = df[(df['Category'] == i) & (df['Investment'] == j) & (df['Currency'] == k)]
                if temp_df.empty == True:
                    continue

                # Get minimum value with the corresponding year
                min_val, min_yr = get_min_value(temp_df)

                # Get maximum value with the corresponding year
                max_val, max_yr = get_max_value(temp_df)

                # Get mean value
                mean_val = get_mean_value(temp_df)
                temp_arr = [i, j, k, min_val, min_yr, max_val, max_yr, mean_val]
                tab.append(temp_arr)
    return tab


# Convert array to dataframe
def arr_to_df(arr, columns):
    tab_df = pd.DataFrame(arr, columns=columns)
    return tab_df


# Empty graph database
def empty_database():
    query = f'''MATCH (n) DETACH DELETE n'''
    graph.run(query)


# Write to neo4j database
def write_to_graphdb(df):
    dataset_length = len(df)
    for i in range(dataset_length):
        if pd.isna(df['Value'][i]) == True:
            # Replacing missing values with NaN
            amount = 'NaN'
        else:
            # Converting value column to integer
            amount = int(df['Value'][i])

        # Define the nodes
        new_investment = Node('Investment', name=df['Investment'][i])
        new_currency = Node('Currency', code=df['Currency'][i])
        new_value = Node('Value', amount=amount, id=i, multiplier='million')

        # Define the relationship
        investment_currency = Relationship(new_investment, 'IN', new_currency, type=df['Category'][i])
        currency_value = Relationship(new_currency, 'EQUIVALENCE OF', new_value, year=int(df['Date'][i]))

        # Create the nodes and relationships
        graph.merge(new_investment, 'Investment', 'name')
        graph.merge(new_currency, 'Currency', 'code')
        graph.merge(investment_currency)
        graph.merge(new_value, 'Value', 'amount')
        graph.merge(currency_value)


# Retrieve information from Neo4j database
def retrive_data(limit):
    result = session.run(f"MATCH (s)-[r0]->(t)-[r1]->(u) RETURN s,r0,t,r1,u LIMIT {limit}")
    return result


def run(df):
    print(f"\n\nPRE PROCESS DATA STATS:\n{'*' * 25}\n\n")
    print(f"SHAPE:{df.shape} \n\n"
          f'DATASET DESCRIPTION:\n{df.describe()} \n\nDATAFRAME:\n{df.tail()} \n\n'
          f'MISSING VALUES:\n{df.isnull().sum()}\n\n')

    deep = Helper.deep_exploration(df)
    print(deep[0][0], deep[0][1])
    print(deep[1])

# Display graph database information
# def display_graph(result):
#     w = GraphWidget(graph=result.graph())
#     w.show()

# Import dataset
import_csv()
df = load_saved_raw_data()
run(df)

# Preview dataset
data_preview(df, 10)