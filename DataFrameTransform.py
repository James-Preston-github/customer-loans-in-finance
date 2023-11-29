import pandas as pd
from scipy import stats
import numpy as np

def adder(x,y):
    return x+y

class DataFrameTransform:
    def __init__(self, df):
        self.df = df
    def impute_missing_values(self, df, columns_to_impute_median=[], columns_to_impute_mean=[]):
        for column in columns_to_impute_median:
            col = df[column]
            median_value = col.median()
            df[column].fillna(median_value, inplace=True)
        for column in columns_to_impute_mean:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        return df

    def dropper(self, df, columns_getting_dropped):
        for column in columns_getting_dropped:
            df.drop(column, axis=1, inplace=True)
        return df

    def columns_to_mean_from_nulls(self, list_of_nulls):
        print(f'str({str(list_of_nulls)} this is the test')
        column_of_means = []
        column_of_medians = [list_of_nulls[x] for x in range(0,len(list_of_nulls)-1)]
        indices = list(input("Which indices of columns do you want to be mean imputed as a single number; i.e. 123 would be 1 and 2 and 3. :"))
        for index in indices:
            column_of_means.append(list_of_nulls[int(index)])
            column_of_medians.remove(list_of_nulls[int(index)])
        return [column_of_means, column_of_medians]
    def log_correction_of_column(self, df, column):
        log_column = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_column
    def boxcox_correction_of_column(self, df, column):
        try:
            boxcox_correction = stats.boxcox(df[column])
            print('worked')
            return pd.Series(boxcox_correction[0])
        except:
            print(f'{column} doesn\'t work')
            return df[column]
    def yeo_correction_of_column(self, df, column):
        try:
            yeo_correction = stats.yeojohnson(df[column])
            yeo_correction = pd.Series(yeo_correction[0])
            return yeo_correction
        except:
            return df[column]
    def log_correcting(self, df, list_of_columns):
        for column in list_of_columns:
            df[column] = df[column].map(lambda i: np.log(i) if i >0 else 0)
        return df
    def boxcox_correcting(self, df, list_of_columns):
        for column in list_of_columns:
            print(f'tried on {column}')
            boxcox_population= stats.boxcox(df[column])
            df[column]= pd.Series(boxcox_population[0])
        return df
    def yeo_correcting(self, df, list_of_columns):
        for column in list_of_columns:
            yeo_correction = stats.yeojohnson(df[column])
            df[column] = pd.Series(yeo_correction[0])
        return df