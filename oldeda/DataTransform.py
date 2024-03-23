import pandas as pd

class DataTransformModified:
    def __init__(self):
        pass
    
    @staticmethod
    def add_to_bool(df, columns_to_bool, addition):
        columns_to_bool.append(addition)
        return columns_to_bool
    
    @staticmethod
    def add_to_cat(df, columns_to_cat, addition):
        columns_to_cat.append(addition)
        return columns_to_cat
    
    @staticmethod
    def add_to_date(df, columns_to_date, addition):
        columns_to_date.append(addition)
        return columns_to_date
    
    @staticmethod
    def convert_to_bool(df, column_values_dict):
        df_copy = df.copy()
        for column, (true_value, false_value) in column_values_dict.items():
            df_copy[column] = df_copy[column].isin([true_value])
        return df_copy
    
    @staticmethod
    def convert_to_cat(df, columns_to_cat):
        df_copy = df.copy()
        df_copy[columns_to_cat] = df_copy[columns_to_cat].astype('category')
        return df_copy
    
    @staticmethod
    def convert_to_date(df, columns_to_date, date_format):
        df_copy = df.copy()
        for column in columns_to_date:
            df_copy[column] = pd.to_datetime(df_copy[column], format=date_format)
        return df_copy
    
    @staticmethod
    def convert_to_int(df, columns_to_int):
        df_copy = df.copy()
        def strip(num):
            halfway = str(num)[:2]
            try:
                return int(halfway)
            except:
                return None
        df_copy[columns_to_int] = df_copy[columns_to_int].map(strip)
        return df_copy
    
    @staticmethod
    def type_df(df):
        print(type(df))
    
    @staticmethod
    def make_into_DataFrame(df):
        return pd.DataFrame(df)
    
    @staticmethod
    def write_to_csv(df, filename):
        df.to_csv(filename)
