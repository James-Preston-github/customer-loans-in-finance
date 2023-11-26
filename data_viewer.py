import pandas as pd
import numpy as np
import numpy as np
from datetime import datetime


csv = pd.read_csv('loan_payments.csv')
df = pd.DataFrame(csv)
csv = pd.read_csv('loan_payments.csv')
df = pd.DataFrame(csv)

date_format = '%b-%Y'

column_to_bool = ['payment_plan']
columns_to_cat = ['verification_status', 'loan_status', 'purpose', ]
columns_to_date = ['last_payment_date', 'next_payment_date', 'last_credit_pull_date', 'earliest_credit_line','issue_date']
columns_to_int = ['term', 'employment_length']

def strip(swi):
    halfway = str(swi)[:2]
    try:
        return int(halfway)
    except:
        return None

class DataTransform:
    def __init__(self, data, columns_to_bool=[], column_values_dict={}, columns_to_cat=[], columns_to_date=[], columns_to_int=[]):
        self.df = data
        self.columns_to_bool = columns_to_bool
        self.columns_to_cat = columns_to_cat
        self.column_values_dict = column_values_dict
        self.columns_to_date = columns_to_date
        self.columns_to_int = columns_to_int
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
    def convert_to_int(self):
        df[columns_to_int] = df[columns_to_int].map(strip)
    def make_into_DataFrame(self):
        return pd.DataFrame(self.df)
    def write_to_csv(self, filename):
        self.df.to_csv(filename)




eda_DataTransform = DataTransform(df, column_to_bool, {column_to_bool[0]: ('y', 'n')}, columns_to_cat, columns_to_date, columns_to_int)
eda_DataTransform.convert_to_bool()
eda_DataTransform.convert_to_cat()
eda_DataTransform.convert_to_date()
eda_DataTransform.convert_to_int()
eda_as_df = eda_DataTransform.make_into_DataFrame()
eda_DataTransform.write_to_csv('DataTransform.csv')


class DataFrameInfo:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.df = pd.DataFrame(self.data)
    
    def column_data_type(self, column_name):
        print(self.df[column_name].dtypes)
    
    def get_column_averages(self, column_name):
        col = self.df[column_name]
        try:
            print(f'''The mean is {col.mean()},
The median is {col.median()},
The standard deviation is {col.std()}''')
        except Exception as e:
            print(f'An error occurred: {e}')
    def get_no_of_cats(self, column_name):
        if self.df[column_name].dtype == "category":
            count = df[column_name].nunique()
            print(count)
            return count
        else:
            print(f'not a category, {df[column_name].dtype}')
    def data_shape(self):
        print(self.df.shape)
    def percent_of_nulls(self, column_name):
        null_count = df[column_name].isnull().sum()
        null_percentage = round(100* null_count / 54231, 2)
        print(f'{null_percentage}%')
        return null_percentage
    def write_to_csv(self, filename):
        self.df.to_csv(filename)
    def impute_missing_values(self, columns_to_impute_median=[], columns_to_impute_mean=[]):
        for column in columns_to_impute_median:
            col = self.df[column]
            median_value = col.median()
            self.df[column].fillna(median_value, inplace=True)
        for column in columns_to_impute_mean:
            mean_value = self.df[column].mean()
            self.df[column].fillna(mean_value, inplace=True)
    



eda = DataFrameInfo('DataTransform.csv')

def column_nulls(column):
    print(f'{column} has {eda.percent_of_nulls(column)} nulls')
    return eda.percent_of_nulls(column)

class Plotter:
    def __init__(self, data, impute_percentage=50):
        self.data = pd.read_csv(data)
        self.df = pd.DataFrame(self.data)
        self.impute_percentage = impute_percentage
    def percent_of_nulls(self):
        total_rows = len(self.df)
        columns_to_drop = []
        columns_to_impute = []
        for column_name, values in self.df.items():
            null_count = values.isnull().sum()
            null_percentage = 100 * null_count / total_rows
            if null_percentage >= self.impute_percentage:
                columns_to_drop.append(column_name)
            elif null_percentage >0:
                columns_to_impute.append(column_name)
                print(f'{column_name} has {round(null_percentage, 2)} null percentage')
        print(f'the columns with over {self.impute_percentage}% nulls are {columns_to_drop}')
        return [columns_to_drop, columns_to_impute]
    def dropper(self, columns_getting_dropped):
        for column in columns_getting_dropped:
            self.df.drop(column, axis=1, inplace=True)
        print('dropped those columns')

    def columns_to_mean_from_nulls(self, list_of_nulls):
        print(list_of_nulls)
        column_of_means = []
        indices = list(input("Which indices of columns do you want to be mean imputed as a single number; i.e. 123 would be 1 and 2 and 3. :"))
        for index in indices:
            column_of_means.append(list_of_nulls[int(index)])
        return column_of_means



analysis = Plotter('DataTransform.csv')
print(analysis.percent_of_nulls()[1])
coldrop = analysis.percent_of_nulls()[0]
analysis.dropper(coldrop)
imputated = (analysis.percent_of_nulls())[1]
print(imputated)
column_of_means = analysis.columns_to_mean_from_nulls(imputated)
column_of_medians =  [column for column in imputated if column not in column_of_means]
eda.impute_missing_values(column_of_means, column_of_medians)
