import zlib
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from data_layer import *
from helper_methods import Helper
import seaborn as sns
import numpy as np
from scipy import stats


class MongoModel:
    def __init__(self):
        self.db_name = "pai_proj_db"
        self.connection_string = 'mongodb://localhost:27017/'
        self.inserted_id = "6590df981b1a9163a90973f1"
        self.processed_ids = []

        # DOWNLOAD DATASET AND  STORE RAW FILES IN MONGODB
        ########################################################

    def run(self):
        # Main functionality of the class goes here
        print("Running data analysis...\n")

        self.store_raw_data()
        self.raw_dataset_stats()
        self.histogram_values()
        self.boxplot_outliers()
        self.preprocess_data()
        self.processed_dataset_stats()
        self.plot_value_overtime()
        self.plot_distribution_by_bankinggroup()
        self.plot_corr_matrix()
        self.plot_time_series()
        self.plot_pi_currency_count()

        # *************************

    # Method triggers file download from the Swiss National Bank, Instantiates MongoDb in the data_layer file
    # and sends raw file for storage.
    def store_raw_data(self):
        csv_url = "https://data.snb.ch/api/warehouse/cube/BSTA.SNB.JAHR_U.BIL.AKT.TOT/data/csv/en"
        coll_name = "raw_csv"
        if input("Do you want to download the dataset (y/n)? ").lower() == "y":
            print("Downloading and Storing Raw Data...")
            # DATA SOURCE csv - Swiss Nation Bank (SNB) Total Assets 1987 - 2022.
            snb_raw_csv = Helper.download_file(csv_url)
            data_io = Helper.compress_if_necessary(snb_raw_csv)  # returns bytes data compressed or not
            db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name=coll_name)
            self.inserted_id = db.savefile_in_db(data_io)  # returns inserted_id

    # ***** DATA EXPLORATION REGION *****
    # ########################################################

    # Function to initiate retrieving data from the MongoDb.
    def load_saved_raw_data(self):
        coll_name = "raw_csv"
        db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name=coll_name)
        d = db.loadfile_from_db()
        # Checking if the retrieved data was compressed before storing. If so, decompresses
        if Helper.is_zlib_compressed(d):
            d = zlib.decompress(d)
        d = d.decode()  # string data was encoded to bytes before storing. Decode it back to string
        d_io = StringIO(d)  # Convert the string data to a file-like obj for pd to be able to read

        raw_df = pd.read_csv(d_io, delimiter=';', skiprows=2)  # Skip 1st two rows (They're non-informative)

        # Renaming the col names from German to English
        raw_df.rename(columns={
            'KONSOLIDIERUNGSSTUFE': 'ConsolidationLevel',
            'INLANDAUSLAND': 'DomesticForeign',
            'WAEHRUNG': 'Currency',
            'BANKENGRUPPE': 'BankingGroup'
        }, inplace=True)
        return raw_df

    # 1. Raw Dataset Overview
    def raw_dataset_stats(self):
        d = self.load_saved_raw_data()

        empty_values = d[d['Value'].isna()]
        d['Date'] = d['Date'].astype(int)

        fully_filled_df = d[(d['Date'] >= 2015) & (d['Date'] <= 2022)]  # Filter for years 2015 to 2022
        print(f'RAW DATA STATS:\n********************* \n\nSHAPE: {d.shape} '
              f'\n\nDATASET DESCRIPTION:\n{d.describe()} \n\nDATAFRAME:\n{d.head()} '
              f'\n\nMISSING VALUES:\n{d.isnull().sum()}\n\n')

        res = Helper.deep_exploration(d)
        print(f"{res[0][0]} \n\n{res[0][1]} \n\n{res[1]}")

    # 2. Histogram to show a positive skew in the dataset
    def histogram_values(self):
        raw_df = self.load_saved_raw_data()
        values = raw_df['Value'].values.reshape(-1, 1)
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=50, color='skyblue', edgecolor='black')
        plt.title('Histogram of Scaled Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    # 3. Boxplot to show potential outliers
    def boxplot_outliers(self):
        raw_df = self.load_saved_raw_data()
        plt.figure(figsize=(10, 6))
        raw_df.boxplot('Value', vert=False)
        plt.title('Boxplot of Value')
        plt.xlabel('Value')
        plt.show()

    # DATA PREPROCESSING
    ##########################################
    def preprocess_data(self):

        # Funtion that instantiates the Mongodb and passes the processed db to the obj for storage.
        def store_processed_data(df):
            print("Processing data...")
            df.to_csv("processed_csv")
            records = df.to_dict(orient='records')  # DataFrame to a list of dictionaries
            db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name='processed_csv')
            return db.saveprocessed_csv_in_db(records, 'snb_processed_csv')

        raw_df = self.load_saved_raw_data()  #  Sends command to load stored Raw detaset from db

        # A. Invoking the onehot_encoding method of the Helper class and passes Categorical Features for convertion. - OneHot Encoding
        df_cleaned = Helper.onehot_encode(raw_df, ['ConsolidationLevel', 'DomesticForeign', 'Currency', 'BankingGroup'])

        # B. Replacing Missing Values with 0 or mean?
        """
        The decision to replace the missing values in the Value column is based on the following:
        Apart from the NaN, each year contains at least a record with 0 in the Value column. 
        Most of the records in each year contain just 5/252 records with NaN.
        """
        df_cleaned['Value'] = df_cleaned['Value'].fillna(0)
        # df_cleaned['Value'].fillna(df_cleaned['Value'].median(), inplace=True)

         # C. Feature Engineering
        df_cleaned['Decade'] = (df_cleaned['Date'] // 10) * 10

        # D.Normalize/Standardize Numerical Data
        df_cleaned = Helper.scale_values(df_cleaned)
        self.processed_ids = store_processed_data(df_cleaned)

    def processed_dataset_stats(self):
        d = self.__load_saved_precessed_data()
        print(f'PROCESSED DATA STATS:\n********************* \n\nSHAPE:\n{d.shape} '
              f'\n\nDATASET DESCRIPTION:\n{d.describe()} \n\nDATAFRAME:\n{d.tail()} '
              f'\n\nMISSING VALUES:\n{d.isnull().sum()}\n\n')


    # VISUALISATION
    ##########################################################

    # 1. Method displays a Scatter plot of Value over time.
    def plot_value_overtime(self):
        print("Plotting value over time...")
        df = self.__load_saved_precessed_data()
        plt.scatter(df['Date'], df['Value'])
        plt.title('Value over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    # 2. Method displays Boxplot of Value distribution by Banking Group
    def plot_distribution_by_bankinggroup(self):
        print("Plotting the distribution by banking groups...")
        df = self.__load_saved_precessed_data()
        df = Helper.onehot_decode(df, 'BankingGroup')
        df.boxplot(column='Value', by='BankingGroup', rot=45)
        plt.title('Value Distribution by Banking Group')
        plt.suptitle('')  # Removes the default title to avoid overlap
        plt.xlabel('Banking Group')
        plt.ylabel('Value')
        plt.show()

    # 3. Method displays the correlation amongst the features
    def plot_corr_matrix(self):
        print("Plotting the correlation matrix...")
        df = self.__load_saved_precessed_data()
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Generate a mask for the upper triangle (due to too many features)
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Generates a custom diverging colormap
        # Drawing heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        # Rotate the labels on the x-axis
        plt.xticks(rotation=45, ha='right')
        plt.show()

    # 4. Method plots an overview of banking behaviour from 1987 to 2022
    def plot_time_series(self):
        print("Plotting Time Series...")

        df = self.__load_saved_precessed_data()
        df.plot(x='Date', y='Value')
        plt.title('Time Series Plot of Value')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    # 6. Method plots a pi chart of the ratio of different currencies
    def plot_pi_currency_count(self):
        print("Plotting Currency counts...")

        df = self.__load_saved_precessed_data()
        df = Helper.onehot_decode(df, 'Currency')
        currency_counts = df['Currency'].value_counts()
        plt.figure(figsize=(8, 8))
        currency_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
        plt.title('Distribution of Currencies')
        plt.ylabel('')  # Hide the y-label as it's not informative for pie charts
        plt.show()

        print(df.head())

    # Method responsible for getting the dataset from the data layer.
    def __load_saved_precessed_data(self):
        db = MongoDB(db_name=self.db_name, cnn_str=self.connection_string, collection_name='processed_csv')
        documents = db.load_all_saved_documents()
        # db.close_connection()
        d = pd.DataFrame(documents)
        return d.drop('_id', axis=1)


if __name__ == '__main__':
    model = MongoModel()
    model.run()


