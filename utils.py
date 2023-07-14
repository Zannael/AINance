import yfinance as yf
import re
import glob
import os.path
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime, timedelta


# Old function to download from Yahoo Finance. We switched to Investing.com
# May turn out to be useful in the future, but for now is useless
def download(title):
    msft = yf.Ticker(title)

    # Remove symbols using regular expression pattern
    title = re.sub(r'[^a-zA-Z]', '', title)

    # get historical market data
    hist = msft.history(period="5y")
    hist = hist.reset_index()
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close']]
    hist['Date'] = hist['Date'].dt.date

    file_path = 'data/raw/' + title + '.csv'  # Specify the file path where you want to save the CSV file
    hist.to_csv(file_path, index=False)  # Save the DataFrame as a CSV file without including the index


# This function is useful to replace the non-default separators in csv files
# Pandas uses "," as default separator, but not all csv files use it, so
# if you have a csv file with another separator, just put sep=c where c
# is the separator utilized.
# It also converts to float string values representing numebrs
# The function also checks for date formats different from Mon dd yyyy and converts them
def clean_separator(sep=";"):
    # Retrieve CSV file paths in the directory
    csv_files = glob.glob('data/raw' + '/*.csv')

    for file in csv_files:
        print(file)
        # Read the CSV file with specified separator
        df = pd.read_csv(file, sep=sep)
        df = df[::-1]
        # Replace separator with commas
        df = df.replace(sep, ',', regex=True)

        # for each column in which we are interested convert the string value to float
        for col_name in df.columns.tolist():
            if col_name in ['Price', 'Open', 'High', 'Low']:
                try: df[col_name] = df[col_name].str.replace(',', '').astype(float)
                except AttributeError: continue

        # takes the first date just to check format
        sample_date = df['Date'][0]

        # tries our default format, if ValueError converts the date to default format
        try: datetime.strptime(sample_date, "%b %d %Y")
        except ValueError: df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%b %d %Y')

        # it save data in a processed folder
        df.to_csv("data/processed/" + os.path.basename(file), index=False)


# This function is used to fill any CSV that does not have values for each day.
# In fact, BTC is a market opened 24/7, while NASDAQ (for example) is opened only
# from Monday to Friday. This causes missing values when puting the two datasets
# side by side, and it's not so good when we have to predict
def fill_dates():
    # Retrieve CSV file paths in the directory
    csv_files = glob.glob('data/processed' + '/*.csv')
    import csv

    # this function, given X and Y, generates values between X and Y
    # with steps equally distant from X, Y and eachother
    def generate_values(X, Y, step):
        if X < Y:
            increment = (Y - X) / (step + 1)
            values = [X + i * increment for i in range(1, step)]
        else:
            increment = (X - Y) / (step + 1)
            values = [X - i * increment for i in range(1, step)]

        return values

    # this function, given a start date and a end date
    # produces all the date objects between the two
    def generate_dates(start_date, end_date):
        missing_dates = []
        current_date = start_date + timedelta(days=1)

        while current_date < end_date:
            missing_dates.append(current_date)
            current_date += timedelta(days=1)

        return missing_dates

    # Loop through each CSV file
    for file in csv_files:
        csv_file = csv.reader(open(file))

        # this dictionary will store pointers to existing dates in the dataset, that have
        # missing dates afterwards. For example, if in the CSV:
        # Date          Col1            Col2            ...
        # Jul 23 2021   ...             ...             ...
        # Jul 26 2021   ...             ...             ...
        # This dictionary will store key-value pairs <k, v>, where:
        # k = Jul 23 2021
        # v = list of missing rows between Jul 23 2021 and Jul 26 2021, so
        #       Jul 24 2021 -> values, Jul 25 2021 -> values
        date_keys = {}
        columns = next(csv_file)    # name of columns, next reads a line and returns it, by moving pointer to the next
        prec_row = next(csv_file)   # we are in the 2nd line, and we read the first row, that we call precedent row

        for row in csv_file:    # remember that, due to next being used 2 times, this actually starts from the 3rd row
            date = datetime.strptime(row[0], "%b %d %Y")    # date of this line
            prec_date = datetime.strptime(prec_row[0], "%b %d %Y")  # date of precedent line
            n_subs = (date - prec_date).days    # difference of dates (should be 1 if no dates are missing)

            if n_subs > 1:  # if > 1, this means that between two rows there are missing lines
                new_dates = generate_dates(prec_date, date)  # we generate these missing dates
                new_values = []  # this will contain missing rows generated values
                for i in range(1, len(columns) - 2):    # for each column but the last two (not interested in them)
                    # we generate a "fake" version of values by taking the ends (estremi) of the range
                    # between the two rows
                    mock_prices = generate_values(float(prec_row[i]), float(row[i]), n_subs)
                    new_values.append(mock_prices)

                # syntactic Python sugar, basically new_values is build like that:
                # [
                #   [col1_value_row1, col1_value_row2, ..., col1_value_rown],
                #   [col2_value_row1, col2_value_row2, ..., col2_value_rown]
                # ]
                # So, each array is all the values of a specific column for all the missing rows
                # But instead we want something like:
                # [
                #   [col1_value_row1, col2_value_row1, col3_value_row1, ..., coln_value_row1],
                #   [col1_value_row2, col2_value_row2, col3_value_row2, ..., coln_value_row2],
                # ]
                # Where each array is exactly the "fake" row that we want to add to the dataset
                # Luckily, Python is a super language
                # We add to the head of each sub array also the dates, in order to have the full row nice and ready!
                rows = [[new_dates[i].strftime("%b %d %Y")] + list(list(zip(*new_values))[i]) + ['', ''] for i in range(len(new_dates))]

                # in the dictionary initialized before the for loop we make the date point to the missing values
                date_keys[prec_date.strftime("%b %d %Y")] = rows

            # just updating the previous row as the current row at the end of the loop
            prec_row = row

        # we open the file again, now as a pandas dataframe
        df = pd.read_csv(file)

        # loop over the dictionary made in the for loop
        for date, rows in date_keys.items():
            row_index = df[df['Date'] == date].index[0]  # thisis just the row number of that date

            # Insert rows at the specified index
            to_add = pd.DataFrame(rows)  # convert our "fake" rows to a dataframe
            to_add.columns = columns    # give these rows columns names
            # we just say df = df[:row_index] + our_rows + df[row_index + 1:]
            df = pd.concat([df.loc[:row_index], to_add, df.loc[row_index + 1:]], ignore_index=True)

        # now this super files are saved to cleaned/
        df.to_csv("data/cleaned/" + os.path.basename(file), index=False, float_format='%.4f')


def get_dataset():
    # Set the directory path where the CSV files are located
    directory_path = 'data/cleaned'

    # Retrieve CSV file paths in the directory
    csv_files = glob.glob(directory_path + '/*.csv')

    dataframes = []
    names = []

    for file in [f for f in csv_files if os.path.basename(f).split(".")[0].isupper()]:
        df = pd.read_csv(file)
        filename = file.split('/')[-1].split('.')[0]  # Extract the filename without extension
        mapping = {col: col + '_' + filename if col != 'Date' else 'Date' for col in df.columns}
        df = df.rename(columns=mapping)
        print(df.columns.tolist())
        df = df.filter(regex=r'^(Price\w*|Date)$')
        dataframes.append(df)
        names.append(filename)

    # Initialize the merged DataFrame with the first DataFrame
    merged_df = dataframes[0]

    # Iterate over the remaining DataFrames and merge them with the merged_df
    for i in range(1, len(dataframes)): merged_df = pd.merge(merged_df, dataframes[i], on='Date', how='inner')

    merged_df.to_csv("data/full.csv", index=False, float_format='%.4f')


# Given a number of days, this function shifts the dataset of days
# It is useful to make a model that predicts BTC <days> days before
def shift_dataset(path, days=7):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)

    df['Label'] = ''
    df.loc[df['Price_BTCUSD'] < df['Price_BTCUSD'].shift(-1), 'Label'] = 'Buy'
    df.loc[df['Price_BTCUSD'] >= df['Price_BTCUSD'].shift(-1), 'Label'] = 'Sell'
    df['Label'] = df['Label'].shift(-1)

    # This line creates a new column in the DataFrame called 'Old_Price_BTCUSD'.
    # The values in this new column are obtained by copying the values in the
    # 'Price_BTCUSD'.
    df['Old_Price_BTCUSD'] = df['Price_BTCUSD']

    # Here we shift the column that we have to predict by <days> rows downwards. In other words, it takes the values
    # from the 'Price_BTCUSD' column <days> rows ago and reassigns them to the same column.
    # In this way each row is:
    # to_predict_value x1 x2 ... xn
    # where x1 x2 xn are from day X and to_predict_value is from day X + <days>
    df['Price_BTCUSD'] = df['Price_BTCUSD'].shift(-days)

    # # Don't need it, but I keep it here just in case
    # # Drop rows with NaN values at the bottom
    # df = df.dropna()
    # df = df.reset_index(drop=True)

    # This is the final csv file that will be used for NN data pre-processing
    df.to_csv("data/clean.csv", index=False, float_format='%.4f')

    return df


# This function prepares data for NN input. Reshapes for LSTM, splits
# for train validation and testing and so on
def data_preparation(df):
    # Split the data into features (X) and target variable (y)
    # Convert 'Date' column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Define start and end dates for the subset
    start_date = '2014-01-01'
    # I keep a whole month of further testing data to be sure NN never saw these
    # Besides, this dataset will be shuffled, so is useful to keep some never-seen ordered data
    # to test the network outputs with a trading strategy
    end_date = '2023-07-15'

    # Subset the dataframe based on the date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # # Shuffle the dataset (meh, I don't use it now because the NN has to learn a time pattern)
    # df = df.sample(frac=1, random_state=123)

    # We split between features and target variable
    X = df[['Old_Price_BTCUSD', 'volume']]  # , 'Price_IXIC', 'Price_MSTR', 'Price_NVDA', 'Price_ICE', 'Price_JPM']]
    y = df['Price_BTCUSD']

    # this will contain "Buy" "Sell" labels that we DON'T use now but
    # can be useful if we want to do classification someday
    class_y = df['Label']

    # Encode the target classification variable with 0 1
    # still, NOT USED
    label_encoder = LabelEncoder()
    class_y_encoded = label_encoder.fit_transform(class_y)

    # Normalize the feature data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the train-test split ratio
    test_size = 0.05

    # Split the data into training and testing sets
    data = list(zip(X_scaled, y))  # Combine features and target into a single list
    class_data = list(zip(X_scaled, class_y_encoded))

    # Calculate the number of samples for the testing set
    test_samples = int(test_size * len(data))
    class_test_samples = int(test_size * len(class_data))

    # Split the data
    X_train, y_train = zip(*data[:-test_samples])  # Training set
    X_test, y_test = zip(*data[-test_samples:])  # Testing set
    _, class_y_train = zip(*class_data[:-class_test_samples])
    _, class_y_test = zip(*class_data[-class_test_samples:])

    # Define the validation split ratio
    val_size = 0.15

    # Calculate the number of samples for the validation set
    val_samples = int(val_size * len(X_train))

    # Split the training set into training and validation sets
    X_train, y_train = list(X_train), list(y_train)  # Convert back to lists
    X_val, y_val = X_train[:val_samples], y_train[:val_samples]  # Validation set
    X_train, y_train = X_train[val_samples:], y_train[val_samples:]  # Updated training set

    # Split the training set into training and validation sets
    _, class_y_train = list(X_train), list(class_y_train)  # Convert back to lists
    _, class_y_val = X_train[:val_samples], class_y_train[:val_samples]  # Validation set
    _, class_y_train = X_train[val_samples:], class_y_train[val_samples:]  # Updated training set

    # Convert training, validation, and testing sets to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    class_y_train = np.array(class_y_train)
    class_y_val = np.array(class_y_val)
    class_y_test = np.array(class_y_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    train_X_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    valid_X_reshaped = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    test_X_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return df, X_train, y_train, X_val, y_val, X_test, y_test, train_X_reshaped, valid_X_reshaped, test_X_reshaped
