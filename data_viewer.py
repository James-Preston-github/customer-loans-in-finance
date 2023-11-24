import pandas as pd
from datetime import datetime


df = pd.read_csv('loan_payments.csv')
df = pd.DataFrame(df)

date_format = '%b-%Y'

column_to_bool = ['payment_plan']
columns_to_cat = ['verification_status', 'loan_status', 'purpose', ]
columns_to_date = ['last_payment_date', 'next_payment_date', 'last_credit_pull_date', 'earliest_credit_line','issue_date']
column_to_check = 'application_type'
print(df[columns_to_date])
class DataTransform:
    def __init__(self, data, columns_to_bool=[], column_values_dict={}, columns_to_cat=[], columns_to_date=[]):
        self.df = data
        self.columns_to_bool = columns_to_bool
        self.columns_to_cat = columns_to_cat
        self.column_values_dict = column_values_dict
        self.columns_to_date = columns_to_date
    def add_to_bool(self, addition):
        self.columns_to_bool.append(addition)
        return self.columns_to_bool
    def add_to_cat(self, addition):
        self.columns_to_cat.append(addition)
        return self.columns_to_cat
    def add_to_date(self, addition):
        self.columns_to_date.append(addition)
        return self.columns_to_date
    def convert_to_bool(self):
        for column, (true_value, false_value) in self.column_values_dict.items():
            self.df[column] = self.df[column].isin([true_value])
    def convert_to_cat(self):
        df[columns_to_cat] = df[columns_to_cat].astype('category')
    def convert_to_date(self):
        for column in columns_to_date:
            df[column] = pd.to_datetime(df[column], format=date_format)




eda = DataTransform(df, column_to_bool, {column_to_bool[0]: ('y', 'n')}, columns_to_cat)
eda.convert_to_bool()
eda.convert_to_cat()
eda.convert_to_date()
