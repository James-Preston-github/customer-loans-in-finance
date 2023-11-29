import pandas as pd


def strip(num):
    halfway = str(num)[:2]
    try:
        return int(halfway)
    except:
        return None

class DataTransform:
    def __init__(self, df):
        self.df = df
    def add_to_bool(self, columns_to_bool, addition):
        columns_to_bool.append(addition)
        return columns_to_bool
    def add_to_cat(self, columns_to_cat, addition):
        columns_to_cat.append(addition)
        return columns_to_cat
    def add_to_date(self, columns_to_date, addition):
        columns_to_date.append(addition)
        return columns_to_date
    def convert_to_bool(self, df, column_values_dict):
        for column, (true_value, false_value) in column_values_dict.items():
            df[column] = df[column].isin([true_value])
        return df
    def convert_to_cat(self, df, columns_to_cat):
        df[columns_to_cat] = df[columns_to_cat].astype('category')
    def convert_to_date(self, df, columns_to_date, date_format):
        for column in columns_to_date:
            df[column] = pd.to_datetime(df[column], format=date_format)
        return df
    def convert_to_int(self, df, columns_to_int):
        df[columns_to_int] = df[columns_to_int].map(strip)
        return df
    def type_df(self, df):
        print(type(df))
    def make_into_DataFrame(self, df):
        return pd.DataFrame(df)
    def write_to_csv(self, df, filename):
        df.to_csv(filename)
