import pandas as pd


class DataFrameInfo:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
    
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
            count = self.df[column_name].nunique()
            print(count)
            return count
        else:
            print(f'not a category, {self.df[column_name].dtype}')
    
    def data_shape(self):
        print(self.df.shape)
    
    def percent_of_nulls(self, column_name):
        null_count = self.df[column_name].isnull().sum()
        null_percentage = round(100* null_count / 54231, 2)
        print(f'{null_percentage}%')
        return null_percentage
    
    def write_to_csv(self, filename):
        self.df.to_csv(filename)

    def column_percent(self, column_name_num, column_name_den):    
        target_column_sum = self.df[column_name_num].sum()
        total_column_sum = self.df[column_name_den].sum() 

        percentage = (target_column_sum / total_column_sum) * 100 
        return percentage
