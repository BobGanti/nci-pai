Analysis of Swiss National Bank Strategies
Overview
This repository houses a Python project designed to investigate the strategies employed by the Swiss National Bank, which contributed to its resilience during global financial crises. The project fetches, stores, and processes data from the Swiss National Bank, incorporating analysis and visualization components within both MongoDB and MySQL routes.
Project Structure
•	data_layer.py: Manages MongoDB connections and operations for data storage and retrieval.
•	helper_methods.py: Provides helper functions for data processing and analysis.
•	mongo_model.py: Specifics of data handling and analysis using MongoDB.
•	mysql_model.py: Specifics of data handling and analysis using MySQL.
•	nci-pfai-taba-main/: Contains all MySQL route implementations, including data handling, analysis, and visualization.
•	processed_csv/: Folder designated for storing CSV outputs of the preprocessed data.
Setup
Ensure you have Python installed, along with the necessary libraries:
•	pandas
•	pymongo
•	sqlalchemy
Install these using pip:
pip install pandas pymongo sqlalchemy 
Running the Project
To explore the Swiss National Bank's strategies through data:
1.	MongoDB Route:
•	Data is fetched from the Swiss National Bank and stored in MongoDB. Large data sets are compressed prior to storage.
•	Raw data is retrieved, preprocessed, and the results are stored in MongoDB. Each processed row is stored as a separate document and also saved in processed_csv/ as processed_data.csv.
•	Analyse and visualize the preprocessed data directly from MongoDB.
2.	MySQL Route:
•	Refer to the nci-pfai-taba-main/ folder for scripts and documentation related to data handling, analysis, and visualization using MySQL.
Objectives
The primary aim is to analyse the effective strategies of the Swiss National Bank during financial downturns. This involves:
•	Analysing financial data trends.
•	Visualizing key metrics that highlight the bank's operational strategies.
•	Assessing the impact of these strategies on the bank's stability and performance during crises.
