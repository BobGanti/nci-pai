import requests
import zlib
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datalayer import *
from HelperMethods import Helper
import seaborn as sns
import numpy as np


# DOWNLOAD DATASET FROM ONLINE SOURCE
###############################################
"""  
DATA SOURCE 1
csv - Child Abuse Referrals - Number that had a Preliminary Enquiry 2023
"""
# csv_url ="https://datacatalog.tusla.ie/dataset/f982f231-f443-4629-8627-d0fd083b0178/resource/88c87124-3b9d-40f5-a3c7-673315e6960f/download/child-abuse-referrals-number-that-had-a-preliminary-enquiry-2023.csv"
# child_abuse_raw_csv = Helper.download_file(csv_url)

"""
DATA SOURCE 2
 json - Electric Vehicle Population Data - json file
State of Washington — This dataset shows the Battery Electric Vehicles (BEVs) and Plug-in Hybrid 
Electric Vehicles (PHEVs) that are currently registered through Washington State Department...
"""
# js_url = "https://data.wa.gov/api/views/f6w7-q2d2/rows.json?accessType=DOWNLOAD"
# evehicle_raw_js = download_file(js_url)

"""
DATA SOURCE 3
csv - Swiss Nation Bank (SNB) Banking Record.
"""
csv_url = "https://data.snb.ch/api/warehouse/cube/BSTA.SNB.JAHR_U.BIL.AKT.TOT/data/csv/en"
snb_raw_csv = Helper.download_file(csv_url)

# STORE RAW FILES IN MONGODB
########################################################
connection_string = 'mongodb+srv://bobganti:8TpFIjetw3BczFvt@nci-projects-db.t2mhes4.mongodb.net/'
raw_csv = snb_raw_csv
db_name = "pai_proj_db"
data_io_name = "snb_raw_csv"
rcollection_name = "raw_csv"
def store_raw_data():
    data_io = Helper.compress_if_necessary(raw_csv)  # returns bytes data compressed or not
    db = MongoDb(db_name=db_name, cnn_str=connection_string, collection_name=rcollection_name)
    return db.savefile_in_db(data_io, data_io_name)
obj_id = store_raw_data()


# #  DATA EXPLORATION
# ########################################################
#
# Function to initiate retrieving data from the MongoDb.
def load_saved_raw_data(obj_id):
    db = MongoDb(db_name=db_name, cnn_str=connection_string, collection_name=rcollection_name)
    d = db.loadfile_from_db(obj_id)
    # Checking if the retrieved data was compressed before storing. If so, decompresses
    if Helper.is_zlib_compressed(d):
        d = zlib.decompress(d)
    d = d.decode()  # string data was encoded to bytes before storing. Decode it back to string
    return StringIO(d)  # Convert the string data to a file-like obj for pd to be able to read

# Skip 1st two rows (They're non-informative)
# raw_df = pd.read_csv(load_saved_raw_data("6583cb0e3cff4f59863376fc"), delimiter=';', skiprows=2)

# Renaming the col names from German to English
col_dict = {
    'KONSOLIDIERUNGSSTUFE': 'ConsolidationLevel',
    'INLANDAUSLAND': 'DomesticForeign',
    'WAEHRUNG': 'Currency',
    'BANKENGRUPPE': 'BankingGroup'
}
# raw_df.rename(columns=col_dict, inplace=True)

# 1. Histogram to show a positive shew in the dataset
def histogram(d):
    scaled_values = d['Value'].values.reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    plt.hist(scaled_values, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Scaled Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
# histogram(raw_df)


# 2. Boxplot to show potential outliers
def boxplot_outliers(d):
    plt.figure(figsize=(10, 6))
    d.boxplot('Value', vert=False)
    plt.title('Boxplot of Value')
    plt.xlabel('Value')
    plt.show()
# boxplot_outliers(raw_df)

# DATA PREPROCESSING
##########################################
def preprocess_data(d):
    # 1. Convert Categorical Data to Numeric Format - OneHot Encoding
    d = Helper.onehot_encode(d, ['ConsolidationLevel', 'DomesticForeign', 'Currency', 'BankingGroup'])

    # 2. Replacing Missing Values
    d['Value'].fillna(d['Value'].median(), inplace=True)

     # 3. Feature Engineering
    d['Decade'] = (d['Date'] // 10) * 10

    # 4.Normalize/Standardize Numerical Data
    d = Helper.scale_values(d)
    return d
# df_to_store = preprocess_data(raw_df)


def store_processec_data(d):
    # Convert the DataFrame to a list of dictionaries
    records = d.to_dict(orient='records')
    db = MongoDb(db_name=db_name, cnn_str=connection_string, collection_name='processed_csv')
    ids = db.saveprocessed_in_db(records, 'snb_processed_csv')
    return ids
# ids = store_processec_data(df_to_store)


# VISUALISATION
##########################################################
def load_saved_precessed_data():
    db = MongoDb(db_name=db_name, cnn_str=connection_string, collection_name='processed_csv')
    documents = db.load_all_saved_documents()
    # db.close_connection()
    d = pd.DataFrame(documents)
    new_d = d.drop('_id', axis=1)
    return new_d
# df_loaded = load_saved_precessed_data()



# 1. Scatter plot of Value over time.
def value_overtime(d):
    plt.scatter(d['Date'], d['Value'])
    plt.title('Value over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
# value_overtime(df_loaded)


# 2. Boxplot of Value distribution by Banking Group
def distribution_by_bankinggroup(d, col):
    d = Helper.onehot_decode(d, col)
    d.boxplot(column='Value', by='BankingGroup', rot=45)
    plt.title('Value Distribution by Banking Group')
    plt.suptitle('')  # Removes the default title to avoid overlap
    plt.xlabel('Banking Group')
    plt.ylabel('Value')
    plt.show()
# distribution_by_bankinggroup(df_loaded, 'BankingGroup')


# Calculate the correlation matrix
def corr_matrix(d):
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # Rotate the labels on the x-axis
    plt.xticks(rotation=45, ha='right')
    plt.show()
# corr_matrix(df_loaded)


def time_series(d):
    d.plot(x='Date', y='Value')
    plt.title('Time Series Plot of Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
# time_series(df_loaded)


def pi_currency_count(d, col):
    d = Helper.onehot_decode(d, col)
    currency_counts = d['Currency'].value_counts()
    plt.figure(figsize=(8, 8))
    currency_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Distribution of Currencies')
    plt.ylabel('')  # Hide the y-label as it's not informative for pie charts
    plt.show()
# pi_currency_count(df_loaded, 'Currency')


# # 5. Split the Data into Features and Labels
# X = df.drop('Value', axis=1)  # Features
# y = df['Value']               # Labels
#
# # 6. Splitting the dataset into 60% train, 20% validation, 20% test
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


