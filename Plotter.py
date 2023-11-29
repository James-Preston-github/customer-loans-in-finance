import pandas as pd
import numpy as np




class Plotter:
    def __init__(self, df):
        self.df = df
    def percent_of_nulls(self, df, impute_percentage=50):
        total_rows = len(df)
        columns_to_drop = []
        columns_to_impute = []
        for column_name, values in df.items():
            null_count = values.isnull().sum()
            null_percentage = 100 * null_count / total_rows
            if null_percentage >= impute_percentage:
                columns_to_drop.append(column_name)
            elif null_percentage >0 and pd.api.types.is_numeric_dtype(df[column_name]):
                columns_to_impute.append(column_name)
                print(f'{column_name} has {round(null_percentage, 2)} null percentage')
        print(f'the columns with over {impute_percentage}% nulls are {columns_to_drop}')
        return [columns_to_drop, columns_to_impute]
    def skew_column_maker(self, df, columns_to_check_for_skew):
        columns_with_skew = []
        for column in columns_to_check_for_skew:
            skew = df[column].skew()
            if abs(skew) >= 1:
                columns_with_skew.append(column)
            else:
                pass
        return columns_with_skew
    def skew_viewer(self, df, column):
        return df[column].skew()
    def log_skew_corrector(self, df, column):
        log_population = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_population

x=2
'''    def outlier_viewer(self, df, column):
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=df[column], color='lightgreen', showfliers=True)
        sns.swarmplot(y=df[column], color='black', size=5)
        plt.title('Box plot with scatter points of floor plan area')
        plt.show()
    def boxplot(self, df, column_name):
        sns.boxplot(x=df[column_name])
        plt.title(f'Boxplot for {column_name}')
        plt.show()
    def swarmplot(self, df, column_name):
        sns.swarmplot(x=df[column_name])
        plt.title(f'Swarmplot for {column_name}')
        plt.show()'''
