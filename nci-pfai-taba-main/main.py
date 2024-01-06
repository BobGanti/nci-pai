from utils import *
from constants import *
import sys


def get_mysql_connection_details():
    print("Please enter the connection details for your MySQL server.")
    # Prompt the user for MySQL connection details.
    username = input(f"     Enter MySQL username or press enter to default to {default_user_name}: ") or default_user_name
    password = input(f"     Enter MySQL password or press enter to default to none: ") or default_password
    host     = input(f"     Enter MySQL host or press enter to default to {default_host}: ") or default_host
    port     = input(f"     Enter MySQL port or press enter to default to {default_port}: ") or default_port
    #MySQL url
    sql_url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}"
    # MySQL database url
    database_url = f"{sql_url}/{target_database}"
    return sql_url, database_url

def after_data_save(database_url):
    print("Now that data is saved what do you want to do?")
    print("     1. Get and print data from the database")
    print("     2. Exit program")
    print("     3. Process data and print")

    operation_choice = input("Enter your choice (1, 2 or 3): ")

    while operation_choice not in ('1', '2', '3'):
        print("Invalid choice. Please enter 1 or 2.")
        operation_choice = input("Enter your choice (1 or 2): ")

    if operation_choice == '1':
        get_data_from_sql(database_url)
    elif operation_choice == '2':
        print("Exiting program...")
        sys.exit()
    elif operation_choice == '3':
        process_data(database_url)

def main():
    print("Welcome to the CSV to SQL converter.")
    print("This program will download a CSV file from a URL and save it to a SQL database.\n")
    print("You can choose between using MySQL or SQLite databases.")
    print("     1. MySQL: This option requires you to have a MySQL server running.")
    print("     2. SQLite: This option will create a SQLite database file in the same directory as this script.")

    choice = input("Enter your choice (1 or 2): ")

    while choice not in ('1', '2'):
        print("Invalid choice. Please enter 1 or 2.")
        choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # MySQL
        while True:
            sql_url, database_url = get_mysql_connection_details()

            # Test MySQL connection
            if prepare_my_sql(sql_url, database_url):
                data_er, data_ft = extract_csv_data()
                load_data_to_sql(data_er, database_url, table_name_er)
                load_data_to_sql(data_ft, database_url, table_name_ft)
                print("CSV file saved to MySQL database.\n")
                after_data_save(database_url)
                break
            else:
                print("Connection to MySQL failed.")
                retry = input("Do you want to retry with different MySQL connection details? (yes/no): ").lower()
                if retry != 'yes':
                    print("Switching to SQLite.")
                    database_url = sqlite_database_url
                    data_er, data_ft = extract_csv_data()
                    load_data_to_sql(data_er, database_url, table_name_er)
                    load_data_to_sql(data_ft, database_url, table_name_ft)
                    print("CSV file saved to SQLite database.\n")
                    after_data_save(database_url)
                    break
    elif choice == '2':
        # SQLite
        database_url = sqlite_database_url
        data_er, data_ft = extract_csv_data()
        load_data_to_sql(data_er, database_url, table_name_er)
        load_data_to_sql(data_ft, database_url, table_name_ft)

        print("CSV file saved to SQLite database.\n")
        after_data_save(database_url)


if __name__ == "__main__":
    main()