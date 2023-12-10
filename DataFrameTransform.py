import pandas as pd
from scipy import stats
import numpy as np
from pandas.tseries.offsets import DateOffset

def adder(x,y):
    return x+y

class DataFrameTransform:
    def __init__(self, df):
        self.df = pd.DataFrame(df)

    def impute_missing_values(self, columns_to_impute_median=[], columns_to_impute_mean=[]):
        for column in columns_to_impute_median:
            col = self.df[column]
            median_value = col.median()
            self.df[column].fillna(median_value, inplace=True)
        for column in columns_to_impute_mean:
            mean_value = self.df[column].mean()
            self.df[column].fillna(mean_value, inplace=True)
        return self.df

    def dropper(self, columns_getting_dropped):
        for column in columns_getting_dropped:
            self.df.drop(column, axis=1, inplace=True)
        return self.df

    def columns_to_mean_from_nulls(self, list_of_nulls):
        column_of_means = []
        column_of_medians = list_of_nulls.copy()
        print(list_of_nulls)
        print(column_of_medians)
        indices = '024'
        for index in indices:
            column_of_means.append(list_of_nulls[int(index)])
            column_of_medians.remove(list_of_nulls[int(index)])
        return [column_of_means, column_of_medians]
    
    def columns_with_numeric(self):
        numeric_cols = []
        for column in self.df.columns.tolist():
            if pd.api.types.is_numeric_dtype(self.df[column]):
                numeric_cols.append(column)
            else:
                pass
        return numeric_cols

    def log_correction_of_column(self, column):
        log_column = self.df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_column
    
    def boxcox_correction_of_column(self, column):
        try:
            boxcox_correction = stats.boxcox(self.df[column])
            return pd.Series(boxcox_correction[0])
        except:
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
        return self.df
    
    def boxcox_correcting(self, list_of_columns):
        for column in list_of_columns:
            boxcox_population= stats.boxcox(self.df[column])
            self.df[column]= pd.Series(boxcox_population[0])
        return self.df
    
    def yeo_correcting(self, list_of_columns):
        for column in list_of_columns:
            yeo_correction = stats.yeojohnson(self.df[column])
            self.df[column] = pd.Series(yeo_correction[0])
        return self.df
    

    def method_for_correction(self, columns_with_skew):
        columns_to_boxcox_adjust=[]
        columns_to_log_adjust=[]
        columns_to_yeo_adjust=[]
        for column in columns_with_skew:
            column_skew_initial = self.df[column].skew()
            log_skew = (self.log_correction_of_column(column)).skew()
            boxcox_skew = (self.boxcox_correction_of_column(column)).skew()
            yeo_skew = self.yeo_correction_of_column(column).skew()
            list_of_skews = [abs(column_skew_initial), abs(log_skew), abs(boxcox_skew), abs(yeo_skew)]
            min_skew_value = min(list_of_skews)
            if min_skew_value == abs(column_skew_initial):
                pass
            elif min_skew_value == abs(log_skew):
                columns_to_log_adjust.append(column)
            elif min_skew_value == abs(boxcox_skew):
                columns_to_boxcox_adjust.append(column)
            elif min_skew_value == abs(yeo_skew):
                columns_to_yeo_adjust.append(column)
        return [columns_to_log_adjust, columns_to_boxcox_adjust, columns_to_yeo_adjust]
    
    def zscore_dropper(self, df_zscore, columns, zscore_threshold=3):
        for column in columns:
            zscore_col = df_zscore[column]
            high_zscore_indices = zscore_col[abs(zscore_col) > zscore_threshold].index
            self.df.loc[high_zscore_indices, column] = None

    def predictions(self,prediction_time):
        collections_df = self.df.copy() # Create copy of the dataframe.

        collections_df['final_payment_date'] = (collections_df['last_payment_date'].max())
        
        def calculate_term_end(row):
            if row['term'] == 36: # In 36 month terms
                return row['issue_date'] + DateOffset(months=36)# Term end will be 36 months after issue date.
            elif row['term'] == 60: # In 60 month terms
                return row['issue_date'] + DateOffset(months=60)

        # Apply the function to create the new 'term_end_date' column
        collections_df['term_end_date'] = collections_df.apply(calculate_term_end, axis=1)

        collections_df['term_end_date'] = pd.to_datetime(collections_df['term_end_date'])
        collections_df['final_payment_date'] = pd.to_datetime(collections_df['final_payment_date'])

        collections_df['mths_left'] = (collections_df['term_end_date'] - collections_df['final_payment_date']).apply(lambda x: x.days // 30)

        collections_df = collections_df[collections_df['mths_left']>0] # filter in only current loans.

        def calculate_collections(row): # Define function to sum collections over projection period.
            if row['mths_left'] >= prediction_time: # If months left in term are equal to or greater than projection period.
                return row['instalment'] * prediction_time #  projection period * Installments.
            elif row['mths_left'] < prediction_time: # If less than projection period months left in term.
                return row['instalment'] * row['mths_left'] # number of months left * installments.

        collections_df['collections_over_period'] = collections_df.apply(calculate_collections, axis=1) # Apply method to each row to get total collections in projected perid.

        collection_sum = collections_df['collections_over_period'].sum()
        total_loan = collections_df['loan_amount'].sum()
        total_loan_left = total_loan - collections_df['total_payment'].sum()

        return {'total_collections': collection_sum, 'total_loan': total_loan, 'total_loan_outstanding': total_loan_left}
    