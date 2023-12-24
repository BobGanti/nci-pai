# Swiss Bank Data Analysis
This project involves downloading, processing, and analyzing financial datasets from the Swiss National Bank. 
The data reflects various banking metrics and is used to perform exploratory data analysis, preprocessing for machine learning, and data visualization.

## Description
The project fetches raw CSV data from The Swiss National Bank website, processes the data using Python and pandas, stores it in MongoDB, and then retrieves it for further analysis. 
It involves steps like handling missing values, one-hot encoding categorical variables, and creating visualizations to understand the data better. 
The end goal is to prepare the data for a machine learning task that predicts financial metrics or classifies banking groups.

### Dependencies
- Python 3.x
- pandas for data manipulation
- matplotlib and seaborn for data visualization
- pymongo for interacting with MongoDB
- scikit-learn for machine learning

### Executing program
- Ensure MongoDB is running on your system.
- Run the script to download and process the data (include the file name and any necessary commands or flags).
- Execute other scripts or Jupyter notebooks for further analysis or machine learning.

```bash
python data_processing.py
