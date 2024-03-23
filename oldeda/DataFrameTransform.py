    
import pandas as pd
from scipy import stats
import numpy as np
from pandas.tseries.offsets import DateOffset

def adder(x, y):
    return x + y

class DataFrameTransformModified:
    def __init__(self):
        pass

    @staticmethod
    def impute_missing_values(df, columns_to_impute_median=[], columns_to_impute_mean=[]):
        df_copy = df.copy()
        for column in columns_to_impute_median:
            col = df_copy[column]
            median_value = col.median()
            df_copy[column].fillna(median_value, inplace=True)
        for column in columns_to_impute_mean:
            mean_value = df_copy[column].mean()
            df_copy[column].fillna(mean_value, inplace=True)
        return df_copy

    @staticmethod
    def dropper(df, columns_getting_dropped):
        df_copy = df.copy()
        for column in columns_getting_dropped:
            df_copy.drop(column, axis=1, inplace=True)
        return df_copy

    @staticmethod
    def columns_to_mean_from_nulls(df, list_of_nulls):
        column_of_means = []
        column_of_medians = list_of_nulls.copy()
        indices = '024'
        for index in indices:
            column_of_means.append(list_of_nulls[int(index)])
            column_of_medians.remove(list_of_nulls[int(index)])
        return [column_of_means, column_of_medians]

    @staticmethod
    def columns_with_numeric(df):
        numeric_cols = []
        for column in df.columns.tolist():
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_cols.append(column)
            else:
                pass
        return numeric_cols

    @staticmethod
    def log_correction_of_column(df, column):
        df_copy = df.copy()
        log_column = df_copy[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_column

    @staticmethod
    def boxcox_correction_of_column(df, column):
        df_copy = df.copy()
        try:
            boxcox_correction = stats.boxcox(df_copy[column])
            return pd.Series(boxcox_correction[0])
        except:
            return df_copy[column]

    @staticmethod
    def yeo_correction_of_column(df, column):
        df_copy = df.copy()
        try:
            yeo_correction = stats.yeojohnson(df_copy[column])
            yeo_correction = pd.Series(yeo_correction[0])
            return yeo_correction
        except:
            return df_copy[column]

    @staticmethod
    def log_correcting(df, list_of_columns):
        df_copy = df.copy()
        for column in list_of_columns:
            df_copy[column] = df_copy[column].map(lambda i: np.log(i) if i > 0 else 0)
        return df_copy

    @staticmethod
    def boxcox_correcting(df, list_of_columns):
        df_copy = df.copy()
        for column in list_of_columns:
            boxcox_population = stats.boxcox(df_copy[column])
            df_copy[column] = pd.Series(boxcox_population[0])
        return df_copy

    @staticmethod
    def yeo_correcting(df, list_of_columns):
        df_copy = df.copy()
        for column in list_of_columns:
            yeo_correction = stats.yeojohnson(df_copy[column])
            df_copy[column] = pd.Series(yeo_correction[0])
        return df_copy

    @staticmethod
    def method_for_correction(df, columns_with_skew):
        columns_to_boxcox_adjust = []
        columns_to_log_adjust = []
        columns_to_yeo_adjust = []
        for column in columns_with_skew:
            column_skew_initial = df[column].skew()
            log_skew = (DataFrameTransformModified.log_correction_of_column(df, column)).skew()
            boxcox_skew = (DataFrameTransformModified.boxcox_correction_of_column(df, column)).skew()
            yeo_skew = DataFrameTransformModified.yeo_correction_of_column(df, column).skew()
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

    @staticmethod
    def zscore_dropper(df, df_zscore, columns, zscore_threshold=3):
        df_copy = df.copy()
        for column in columns:
            zscore_col = df_zscore[column]
            high_zscore_indices = zscore_col[abs(zscore_col) > zscore_threshold].index
            df_copy.loc[high_zscore_indices, column] = None

    @staticmethod
    def predictions(df, prediction_time):
        collections_df = df.copy()

        collections_df['final_payment_date'] = (collections_df['last_payment_date'].max())

        def calculate_term_end(row):
            if row['term'] == 36:
                return row['issue_date'] + DateOffset(months=36)
            elif row['term'] == 60:
                return row['issue_date'] + DateOffset(months=60)

        collections_df['term_end_date'] = collections_df.apply(calculate_term_end, axis=1)

        collections_df['term_end_date'] = pd.to_datetime(collections_df['term_end_date'])
        collections_df['final_payment_date'] = pd.to_datetime(collections_df['final_payment_date'])

        collections_df['mths_left'] = (collections_df['term_end_date'] - collections_df['final_payment_date']).apply(
            lambda x: x.days // 30)

        collections_df = collections_df[collections_df['mths_left'] > 0]

        def calculate_collections(row):
            if row['mths_left'] >= prediction_time:
                return row['instalment'] * prediction_time
            elif row['mths_left'] < prediction_time:
                return row['instalment'] * row['mths_left']

        collections_df['collections_over_period'] = collections_df.apply(calculate_collections, axis=1)

        collection_sum = collections_df['collections_over_period'].sum()
        total_loan = collections_df['loan_amount'].sum()
        total_loan_left = total_loan - collections_df['total_payment'].sum()

        return {'total_collections': collection_sum, 'total_loan': total_loan,
                'total_loan_outstanding': total_loan_left}
