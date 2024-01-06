# Data Importer

This Python program allows you to download a CSV file from a URL and save it to a SQL database. You can choose between using MySQL or SQLite databases.

### Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`):
  - `requests`
  - `pandas`
  - `sqlalchemy`
  - `mysql-connector-python`

### Usage
1. Clone the repository to your local machine:
  - `bash`
  - `git clone https://github.com/OmarGutieG/nci-pfai-taba.git`
2. Navigate to the project directory
  - `cd nci-pfai-taba`
3. Create a virtual environment (optional but recommended):
  - `python3 -m venv venv`
4. Activate the virtual environment:
  - On Windows: `venv\Scripts\activate`
  - On macOS/Linux: `source venv/bin/activate`
5. Install the required dependencies:
  - `pip install -r requirements.txt`
6. Run the main program
  - `python main.py`
7. Follow the on-screen instructions to choose the database type (MySQL or SQLite), enter connection details, and perform operations.
8. The program will prompt you to get and print data from the database or exit after saving data.

### Additional Information
- The script uses SQLAlchemy to handle database connections.
- CSV data is downloaded from the following URL: https://data.snb.ch/api/cube/devkua/data/csv/en
- Ensure that you have a MySQL server running if you choose the MySQL option.
- The program will create a SQLite database file in the same directory if you choose the SQLite option.