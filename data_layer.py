from pymongo import MongoClient
import pymysql
import psycopg2
import datetime
from bson import ObjectId


class MongoDb:
    def __init__(self, db_name, cnn_str, collection_name):
        self._database_name = db_name
        self._connection_string = cnn_str
        self._collection_name = collection_name

    # Instantiate the MongoDb client.
    def connect_db(self):
        client = MongoClient(self._connection_string)
        db = client[self._database_name]
        collection = db[self._collection_name]
        return collection

    # Insert the data
    def savefile_in_db(self, data_io, data_io_name):
        collection = self.connect_db()
        document = {'data': data_io, 'name': data_io_name, "created_at": datetime.datetime.now()}
        result = collection.insert_one(document)
        return result.inserted_id

    def saveprocessed_in_db(self, r, name):
        collection = self.connect_db()
        result = collection.insert_many(r)
        return result.inserted_ids

    # Retrieve raw data from mongo db.
    def loadfile_from_db(self, obj_id):
        collection = self.connect_db()
        document = collection.find_one({"_id": ObjectId(obj_id)})
        d = document['data']
        return d

    def load_all_saved_documents(self):
        collection = self.connect_db()
        return list(collection.find())

    def close_connection(self):
        client = MongoClient(self._connection_string)
        client.close()
        return "MongoDB connection closed."


class SQLDb:
    def __init__(self, engine, db_name, cnn_str):
        self.db_engine = engine
        self.db_name = db_name
        self.connection_string = cnn_str
        self.connection = None

    def connect(self):
        try:
            self.connection = pymysql.connect(self.connection_string)
            print("Connection established successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def execute_query(self, query):
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(query)
                self.connection.commit()
                print("Query executed successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")

    def close(self):
        if self.connection:
            self.connection.close()
            print("Connection closed.")

#
# # # db = SQLDatabase(conn_string)
# # # db.connect()
# # # db.execute_query("SELECT * FROM my_table")
# # # db.close()
