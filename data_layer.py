import pandas as pd
from pymongo import MongoClient
import datetime
from sqlalchemy import insert
from sqlalchemy import create_engine

class MongoDB:
    def __init__(self, db_name, cnn_str, collection_name):
        self._database_name = db_name
        self._connection_string = cnn_str
        self._collection_name = collection_name

        # Instantiate the MongoDB client.

    def __connect_db(self):
        client = MongoClient(self._connection_string)
        db = client[self._database_name]
        collection = db[self._collection_name]
        return collection

    # save file to db
    def savefile_in_db(self, data_io):
        collection = self.__connect_db()
        document = {'data': data_io, "created_at": datetime.datetime.now()}
        result = collection.insert_one(document)
        self.__close_connection()
        return result.inserted_id

        # Retrieve raw data from mongo db.

    def loadfile_from_db(self):
        collection = self.__connect_db()
        most_recent_document = collection.find_one(sort=[('_id', -1)])
        d = most_recent_document['data']
        self.__close_connection()
        return d

    def saveprocessed_csv_in_db(self, r, name):
        collection = self.__connect_db()
        result = collection.insert_many(r)
        self.__close_connection()
        return result.inserted_ids

    def load_all_saved_documents(self):
        collection = self.__connect_db()
        documents = list(collection.find())
        self.__close_connection()
        return documents

    def __close_connection(self):
        client = MongoClient(self._connection_string)
        client.close()
        return "MongoDB connection closed."



# *****************************************
# MySQLDB REGION
# *****************************************
# class MySQLDB:
#     def __init__(self, username, password, server_url, database_name):
#         self.server_url = server_url
#         self.database_name = database_name
#         self.username = username
#         self.password = password
#         self.conn_str = conn_str = f'mysql+mysqlconnector://{self.username}:{self.password}@{self.server_url}/{self.database_name}'
#
#     def __get_connection(self):
#         engine = create_engine(self.conn_str)
#         return engine
#
#     def raw_csv_to_db(self, csv, table_name):
#         engine = self.__get_connection()
#         result = df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
#         return result.inserted_id
#
#     def raw_csv_from_db(self, table_name):
#         engine = self.__get_connection()
#
#         df = pd.read_sql_table(table_name, engine)
#         return df
#
#


