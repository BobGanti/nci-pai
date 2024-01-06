from data_layer import MySQLDB
from helper_methods import *
import pandas as pd
from io import StringIO

class MySQLModel:
    def __init__(self):
        pass

    def store_raw_csv(self):

        # Path to your CSV file
        csv_url = "https://data.snb.ch/api/cube/devkua/data/csv/en"
        csv_file = Helper.download_file(csv_url)
        d_io = StringIO(csv_file)
        d = pd.read_csv(d_io, delimiter=';', skiprows=2)

        db = MySQLDB("bobga", "bobgapass", "localhost:3306", "SBN_raw_csv_db")
        return db.raw_csv_to_db(d, "raw_csv")  # returns row


    def raw_dataset_stats(self):
        # d = self.load_saved_raw_data()
        d = self.store_raw_csv()
        print(f"\n\nPRE PROCESS DATA STATS:\n{'*' * 25}\n\n")
        print(f"SHAPE:{d.shape} \n\n"
              f'DATASET DESCRIPTION:\n{d.describe()} \n\nDATAFRAME:\n{d.tail()} \n\n'
              f'MISSING VALUES:\n{d.isnull().sum()}\n\n')
        deep = Helper.deep_exploration(d)
        print(deep[0][0], deep[0][1])
        print(deep[1])

    def run(self):
        print(self.store_raw_csv())
        print(self.raw_dataset_stats())

model = MySQLModel()
cont = model.store_raw_csv()
print(cont)

