import pandas as pd




class DataTransform:
    def __init__(self,df):
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
    def convert_to_bool(self, column_values_dict):
        for column, (true_value, false_value) in column_values_dict.items():
           self.df[column] =self.df[column].isin([true_value])
        return self.df
    def convert_to_cat(self, columns_to_cat):
       self.df[columns_to_cat] =self.df[columns_to_cat].astype('category')
    def convert_to_date(self, columns_to_date, date_format):
        for column in columns_to_date:
           self.df[column] = pd.to_datetime(self.df[column], format=date_format)
        return self.df
    def convert_to_int(self, columns_to_int):
        def strip(num):
            halfway = str(num)[:2]
            try:
                return int(halfway)
            except:
                return None
        self.df[columns_to_int] = self.df[columns_to_int].map(strip)
        return self.df
    def type_df(self):
        print(type(self.df))
    def make_into_DataFrame(self):
        return pd.DataFrame(self.df)
    def write_to_csv(self, filename):
       self.df.to_csv(filename)
