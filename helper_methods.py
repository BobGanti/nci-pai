import requests
import zlib
from sklearn.preprocessing import StandardScaler
import pandas as pd


class Helper:

    # Function checks whether a file is compressed
    @staticmethod
    def is_zlib_compressed(d):
        try:
            zlib.decompress(d)
            return True
        except zlib.error:
            return False

    # Function checks the downloaded file size and compress with 'zlib' if necessary.
    @staticmethod
    def compress_if_necessary(d):

        BSON_SIZE_LIMIT = 16 * 1024 * 1024  # Limit 16MB in Bytes.

        # Check if data is a string and encode it.
        if isinstance(d, str):
            d_bytes = d.encode()
        else:
            d_bytes = d

        if len(d_bytes) >= BSON_SIZE_LIMIT:
            return zlib.compress(d_bytes)
        else:
            return d_bytes

    @staticmethod
    def scale_values(df):
        # Reshaping the 'Value' column as a 2D array
        # Create a StandardScaler object and fit to the data
        scaler = StandardScaler()
        df[['Value']] = scaler.fit_transform(df[['Value']])
        return df

    # # Download data
    @staticmethod
    def download_file(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return "Error."

    @staticmethod
    def date_datetime(df):
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    @staticmethod
    def onehot_encode(df, cols):
        df_enc = pd.get_dummies(df[cols]).astype(int)
        df.drop(cols, axis=1, inplace=True)  # Dropping the original 'Category' column from df
        df_enc.fillna(0, inplace=True)  # Replace NaN in the encoded DataFrame with 0
        return df.join(df_enc)

    @staticmethod
    def onehot_decode(df, col_to_decode):
        cols = [col for col in df.columns if col.startswith(f'{col_to_decode}_')]
        # df.fillna(0, inplace=True)
        df[col_to_decode] = df[cols].idxmax(axis=1)  # returns NaN if the column or row contains all missing values
        df[col_to_decode] = df[col_to_decode].str.replace('BankingGroup_', '')
        return df
