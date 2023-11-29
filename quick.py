import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from scipy import stats
from scipy.stats import yeojohnson, zscore
import matplotlib.pyplot as plt
import plotly.express as px


def calculate_zscores(column):
            if pd.api.types.is_numeric_dtype(column):
                return zscore(column)
            else:
                return column

class DataFrameTransform:
    def __init__(self, df):
        self.df = df
    def impute_missing_values(self, df, columns_to_impute_median=[], columns_to_impute_mean=[]):
        for column in columns_to_impute_median:
            col = df[column]
            median_value = col.median()
            print(f'median is {median_value}')
            df[column].fillna(median_value, inplace=True)
        for column in columns_to_impute_mean:
            mean_value = df[column].mean()
            self.df[column].fillna(mean_value, inplace=True)
        return df

    def dropper(self, columns_getting_dropped):
        for column in columns_getting_dropped:
            self.df.drop(column, axis=1, inplace=True)
        print('dropped those columns')

    def columns_to_mean_from_nulls(self, list_of_nulls):
        print(list_of_nulls)
        column_of_means = []
        column_of_medians = [list_of_nulls[x] for x in range(0,5)]
        indices = list(input("Which indices of columns do you want to be mean imputed as a single number; i.e. 123 would be 1 and 2 and 3. :"))
        for index in indices:
            column_of_means.append(list_of_nulls[int(index)])
            column_of_medians.remove(list_of_nulls[int(index)])
        return [column_of_means, column_of_medians]
    def log_correction_of_column(self,column):
        log_column = self.df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_column
    def boxcox_correction_of_column(self, column):
        try:
            boxcox_correction = stats.boxcox(self.df[column])
            print('worked')
            return pd.Series(boxcox_correction[0])
        except:
            print(f'{column} doesn\'t work')
            return self.df[column]
    def yeo_correction_of_column(self, column):
        try:
            yeo_correction = stats.yeojohnson(self.df[column])
            yeo_correction = pd.Series(yeo_correction[0])
            return yeo_correction
        except:
            return self.df[column]
    def log_correcting(self, list_of_columns):
        for column in list_of_columns:
            self.df[column] = self.df[column].map(lambda i: np.log(i) if i >0 else 0)
    def boxcox_correcting(self, list_of_columns):
        for column in list_of_columns:
            print(f'tried on {column}')
            boxcox_population= stats.boxcox(self.df[column])
            self.df[column]= pd.Series(boxcox_population[0])
    def yeo_correcting(self, list_of_columns):
        for column in list_of_columns:
            yeo_correction = stats.yeojohnson(self.df[column])
            self.df[column] = pd.Series(yeo_correction[0])
    def update_csv(self, filename):
        self.df.to_csv(filename)
    def columns_to_drop(self, corr_df, corr_thresh):
        columns_mask = np.abs(corr_df) >= corr_thresh
        np.fill_diagonal(columns_mask.values, False)
        columns_mask = np.triu(columns_mask)
        row_indices, col_indices = np.where(columns_mask)
        corr_col_list = [[corr_df.columns[row], corr_df.columns[col]] for row, col in zip(row_indices, col_indices)]
        return corr_col_list




class Plotter:
    def __init__(self, df, impute_percentage=50):
        df = df
        self.impute_percentage = impute_percentage
    def percent_of_nulls(self):
        total_rows = len(df)
        columns_to_drop = []
        columns_to_impute = []
        for column_name, values in df.items():
            null_count = values.isnull().sum()
            null_percentage = 100 * null_count / total_rows
            if null_percentage >= self.impute_percentage:
                columns_to_drop.append(column_name)
            elif null_percentage >0 and pd.api.types.is_numeric_dtype(df[column_name]):
                columns_to_impute.append(column_name)
                print(f'{column_name} has {round(null_percentage, 2)} null percentage')
        print(f'the columns with over {self.impute_percentage}% nulls are {columns_to_drop}')
        return [columns_to_drop, columns_to_impute]
    def skew_column_maker(self, columns_to_check_for_skew):
        columns_with_skew = []
        for column in columns_to_check_for_skew:
            skew = df[column].skew()
            if abs(skew) >= 1:
                columns_with_skew.append(column)
            else:
                pass
        return columns_with_skew
    def skew_data_plot(self, column):
        return df[column].skew()
    def log_skew_corrector(self, column):
        log_population = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_population
    def outlier_data_plot(self, column):
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=df[column], color='lightgreen', showfliers=True)
        sns.swarmplot(y=df[column], color='black', size=5)
        plt.title('Box plot with scatter points of floor plan area')
        plt.show()
    def boxplot(self, column_name):
        sns.boxplot(x=df[column_name])
        plt.title(f'Boxplot for {column_name}')
        plt.show()
    def swarmplot(self, column_name):
        sns.swarmplot(x=df[column_name])
        plt.title(f'Swarmplot for {column_name}')
        plt.show()
    def zscore_maker(self):
        zscore_df = df.copy().apply(calculate_zscores)
        return zscore_df
    def calculate_zscores_numeric_columns(self):
        """
        Calculate Z-scores for numerical columns in the DataFrame.

        Returns:
        - zscore_df: DataFrame containing Z-scores for numerical columns.
        """
        zscore_df = df.copy()

        def calculate_zscores(column):
            if pd.api.types.is_numeric_dtype(column):
                # Convert integer columns to float before calculating Z-scores
                if pd.api.types.is_integer_dtype(column):
                    column = column.astype(float)
                return zscore(column)
            else:
                return pd.Series([])

        zscore_df = zscore_df.apply(calculate_zscores)

        return zscore_df

    def zscore_memtion(self, zscore_df, value):
        filtered_df = zscore_df.copy()
        filtered_df[abs(zscore_df) <= value] = pd.NA

        return filtered_df
            

df = pd.read_csv('skew_removed.csv')
'''
skew_and_outlier_checker = Plotter(df)
skew_and_outlier_checker.percent_of_nulls()
zscore_df = skew_and_outlier_checker.zscore_maker()
zscore_result_df = skew_and_outlier_checker.calculate_zscores_numeric_columns()
print(type(zscore_result_df))
zscores_of_mention = skew_and_outlier_checker.zscore_memtion(zscore_result_df, 5)
print(zscores_of_mention.info())
'''

def numeric_columns_correlation(df):
        numeric_columns = df.select_dtypes(include=['number']).columns
        numeric_df = df[numeric_columns]
        correlation_matrix = numeric_df.corr()

        return correlation_matrix, numeric_df
corr_df = numeric_columns_correlation(df)
#px.imshow(corr_df[1].corr(), title="Correlation heatmap of student dataframe")
correlat = DataFrameTransform(df)
removed_correlation_columns = correlat.columns_to_drop(corr_df[0],0.98)

print(removed_correlation_columns)

df.drop('Unnamed: 0', axis=1, inplace=True)

dropper = set([])
for pair in removed_correlation_columns:
    dropper.add(pair[0])
for column in dropper:
    df.drop(column, axis=1, inplace=True)
    print(f'dropping {column}')