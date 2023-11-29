import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from scipy import stats
from scipy.stats import yeojohnson, zscore
import matplotlib.pyplot as plt

csv = pd.read_csv('loan_payments.csv')
df = pd.DataFrame(csv)
csv = pd.read_csv('loan_payments.csv')
df = pd.DataFrame(csv)

date_format = '%b-%Y'

column_to_bool = ['payment_plan', 'inq_last_6mths']
columns_to_cat = ['verification_status', 'loan_status', 'purpose', ]
columns_to_date = ['last_payment_date', 'next_payment_date', 'last_credit_pull_date', 'earliest_credit_line','issue_date']
columns_to_int = ['term', 'employment_length']
columns_that_should_be_skew_viewable = ['loan_amount','funded_amount','funded_amount_inv','term','int_rate','instalment','employment_length','annual_inc','dti','delinq_2yrs','mths_since_last_delinq','mths_since_last_record','open_accounts','total_accounts','out_prncp','out_prncp_inv','total_payment','total_payment_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_payment_amount','mths_since_last_major_derog']
columns_with_skew = []

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
    def type_df(self):
        print(type(self.df))
    def make_into_DataFrame(self):
        return pd.DataFrame(self.df)
    def write_to_csv(self, filename):
        self.df.to_csv(filename)




eda_DataTransform = DataTransform(df, column_to_bool, {column_to_bool[0]: ('y', 'n')}, columns_to_cat, columns_to_date, columns_to_int)
eda_DataTransform.convert_to_bool()
eda_DataTransform.convert_to_cat()
eda_DataTransform.convert_to_date()
eda_DataTransform.convert_to_int()
eda_DataTransform.type_df()
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

def calculate_zscores(column):
            if pd.api.types.is_numeric_dtype(column):
                return zscore(column)
            else:
                return column

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
            elif null_percentage >0 and pd.api.types.is_numeric_dtype(self.df[column_name]):
                columns_to_impute.append(column_name)
                print(f'{column_name} has {round(null_percentage, 2)} null percentage')
        print(f'the columns with over {self.impute_percentage}% nulls are {columns_to_drop}')
        return [columns_to_drop, columns_to_impute]
    def skew_column_maker(self, columns_to_check_for_skew):
        for column in columns_to_check_for_skew:
            skew = self.df[column].skew()
            if abs(skew) >= 1:
                columns_with_skew.append(column)
            else:
                pass
        return columns_with_skew
    def skew_data_plot(self, column):
        return self.df[column].skew()
    def log_skew_corrector(self, column):
        log_population = self.df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_population
    def outlier_data_plot(self, column):
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=self.df[column], color='lightgreen', showfliers=True)
        sns.swarmplot(y=self.df[column], color='black', size=5)
        plt.title('Box plot with scatter points of floor plan area')
        plt.show()
    def boxplot(self, column_name):
        sns.boxplot(x=self.df[column_name])
        plt.title(f'Boxplot for {column_name}')
        plt.show()
    def swarmplot(self, column_name):
        sns.swarmplot(x=self.df[column_name])
        plt.title(f'Swarmplot for {column_name}')
        plt.show()
    def zscore_maker(self):
        zscore_df = self.df.copy().apply(calculate_zscores)
        return zscore_df






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


data_plot = Plotter('DataTransform.csv')
list_of_nulls = data_plot.percent_of_nulls()[1]
list_to_drop = data_plot.percent_of_nulls()[0]
data_frame = DataFrameTransform(data_plot.df)
columns_to_impute = data_frame.columns_to_mean_from_nulls(list_of_nulls)
columns_to_mean = columns_to_impute[0]
columns_to_median = columns_to_impute[1]
data_frame.dropper(list_to_drop)
data_frame = data_frame.impute_missing_values(df, columns_to_median ,columns_to_mean)
data_frame = DataFrameTransform(data_frame)
data_frame.update_csv('DataTransformed.csv')

data_plot = Plotter('DataTransformed.csv')
data_plot.percent_of_nulls()
for column in list_to_drop:
    try:
        columns_that_should_be_skew_viewable.remove(column)
    except:
        pass

plotter = Plotter('DataTransformed.csv', columns_that_should_be_skew_viewable)
data_frame = DataFrameTransform(plotter.df)
columns_with_skew = []
columns_to_log_adjust = []
columns_to_boxcox_adjust = []
columns_to_yeo_adjust = []
yeo_skews = []
columns_with_skew = plotter.skew_column_maker(columns_that_should_be_skew_viewable)
print(columns_with_skew)
print(data_frame.df['annual_inc'].skew())
for column in columns_with_skew:
    column_skew_initial = data_frame.df[column].skew()
    log_skew = (data_frame.log_correction_of_column(column)).skew()
    boxcox_skew = (data_frame.boxcox_correction_of_column(column)).skew()
    yeo_skew = data_frame.yeo_correction_of_column(column).skew()
    if column == 'annual_inc':
        print(yeo_skew)
    list_of_skews = [abs(column_skew_initial), abs(log_skew), abs(boxcox_skew), abs(yeo_skew)]
    min_skew_value = min(list_of_skews)
    if column == 'annual_inc':
        print(min_skew_value)
    if min_skew_value == abs(column_skew_initial):
        pass
    elif min_skew_value == abs(log_skew):
        columns_to_log_adjust.append(column)
        print(f'{column} is log')
    elif min_skew_value == abs(boxcox_skew):
        columns_to_boxcox_adjust.append(column)
        print(f'appended {column}')
    elif min_skew_value == abs(yeo_skew):
        columns_to_yeo_adjust.append(column)
        print(f'{column} is yeo')
        yeo_skews.append([yeo_skew, column_skew_initial])
    #print(f'{log_skew} is the new skew, the old one was {plotter.skew_data_plot(column)}')
#print(f'The columns to log skew adjust are {columns_to_log_adjust}') 
#data_frame.log_correcting(columns_to_log_adjust)
print(columns_to_yeo_adjust, columns_to_log_adjust, columns_to_boxcox_adjust)
data_frame.boxcox_correcting(columns_to_boxcox_adjust)
data_frame.yeo_correcting(columns_to_yeo_adjust)
data_frame.update_csv('skew_removed.csv')
skew_and_outlier_checker = Plotter('skew_removed.csv')
for column in columns_that_should_be_skew_viewable:
    print(skew_and_outlier_checker.skew_data_plot(column), column)
#skew_and_outlier_checker.boxplot('annual_inc')
#skew_and_outlier_checker.swarmplot('')
zscore_df = skew_and_outlier_checker.zscore_maker()
print(zscore_df.describe())


'''for column in columns_to_yeo_adjust:
    t=sns.histplot(data_frame.df[column],label="Skewness: %.2f"%(data_frame.df[column].skew()) )
    t.legend()
    plt.show()'''
'''for column in columns_to_log_adjust:
    print(f'skew is now {data_frame.df[column].skew()}')'''
