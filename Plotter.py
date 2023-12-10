import missingno as msno
import pandas as pd
from scipy.stats import normaltest
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import seaborn as sns
import numpy as np
from scipy.stats import yeojohnson, zscore
from pandas.tseries.offsets import DateOffset




class Plotter:
    def __init__(self, df):
        self.df = df

    def percent_of_nulls(self, impute_percentage=50):
        total_rows = len(self.df)
        columns_to_drop = []
        columns_to_impute = []
        for column_name, values in self.df.items():
            null_count = values.isnull().sum()
            null_percentage = 100 * null_count / total_rows
            if null_percentage >= impute_percentage:
                columns_to_drop.append(column_name)
            elif null_percentage >0 and pd.api.types.is_numeric_dtype(self.df[column_name]):
                columns_to_impute.append(column_name)
                print(f'{column_name} has {round(null_percentage, 2)} null percentage')
        print(f'the columns with over {impute_percentage}% nulls are {columns_to_drop}')
        return [columns_to_drop, columns_to_impute]
    
    def skew_column_maker(self, columns_to_check_for_skew):
        columns_with_skew = []
        for column in columns_to_check_for_skew:
            skew = self.df[column].skew()
            if abs(skew) >= 1:
                columns_with_skew.append(column)
            else:
                pass
        return columns_with_skew
    
    def skew_viewer(self, column):
        return self.df[column].skew()
    
    def log_skew_corrector(self, column):
        log_population = self.df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return log_population
    
    def multi_hist_plot(self, num_cols):
        sns.set(font_scale=0.7)
        f = pd.melt(self.df, value_vars=num_cols)
        g = sns.FacetGrid(f, col="variable", col_wrap=4,
                          sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        pyplot.show()
    
    def qq_plot(self, col):
        self.df.sort_values(by=col, ascending=True)
        qq_plot = qqplot(self.df[col], scale=1, line='q')
        pyplot.show()

    
    def multi_qq_plot(self, cols):
        remainder = 1 if len(cols) % 4 != 0 else 0
        rows = int(len(cols) / 4 + remainder)

        fig, axes = pyplot.subplots(
            ncols=4, nrows=rows, sharex=False, figsize=(6, 3))
        for col, ax in zip(cols, np.ravel(axes)):
            sm.qqplot(self.df[col], line='s', ax=ax, fit=True)
            ax.set_title(f'{col} QQ Plot')
        pyplot.tight_layout()

    def missing_nulls_vis(self):
        msno.matrix(self.df)
        pyplot.show()

    def correlated_vars(self, cols):
        corr = self.df[cols].corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        pyplot.figure(figsize=(10, 8))

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, square=True, linewidths=5,
                    annot=False, cmap=cmap)

        
        pyplot.show()


    def columns_to_drop(self, cols, corr_thresh=0.9):
        corr_df = self.df[cols].corr()
        columns_mask = np.abs(corr_df) >= corr_thresh
        np.fill_diagonal(columns_mask.values, False)
        columns_mask = np.triu(columns_mask)
        row_indices, col_indices = np.where(columns_mask)
        
        
        used_columns = set()
        corr_col_list = []

        for row, col in zip(row_indices, col_indices):
            col1, col2 = corr_df.columns[row], corr_df.columns[col]
            
            
            if col2 not in used_columns:
                corr_col_list.append(col2)
                used_columns.add(col2)
        
        return corr_col_list
    
    def final_payment_date(self):
        self.df['final_date'] = self.df['issue_date'] + DateOffset(months = self.df['term'], axis=1)
        return self.df
    

    def zscore_maker(self):
        def calculate_zscores(entry):
            if pd.api.types.is_numeric_dtype(entry):
                return zscore(entry)
            else:
                return 0
        zscore_df = pd.DataFrame.copy(self.df).apply(calculate_zscores)
        zscore_df = zscore_df.map(lambda y: 0.0 if y<2.0 else y) 
        return zscore_df
    
    
    
class Charts:
    def __init__(self):
        self = self
    def pie_charts(self, labels :list, list_of_data: list, title: str=None):
        pyplot.pie(list_of_data, labels=labels, autopct='%1.1f%%') 
        if title != None:
            pyplot.title(title)
        pyplot.show()
        
    def calc_percent(self, nom, den):
        percent = 100* nom/den
        return percent
    
    def revenue_lost_by_month(self, DataFrame: pd.DataFrame):

        df = DataFrame.copy()

        df['term_completed'] = (df['last_payment_date'] - df['issue_date']) 
        df['term_completed'] = df['term_completed'].dt.days 

        def calculate_term_remaining(row): 
            if row['term'] == 36:
                try:
                    print(f'the num is {row["term_completed"]}, {row["Unnamed: 0"]}')
                    return int(36*30 - int(row['term_completed'])) 
                except:
                    return 36*30
            elif row['term'] == 60:
                try:
                    print(f'the num is {row["term_completed"]}, {row["Unnamed: 0"]}')
                    return int(36*30 - int(row['term_completed'])) 
                except:
                    return 36*30 
        df['term_left'] = df.apply(calculate_term_remaining, axis=1)
        print(df['term_left'].max())
        revenue_lost = [] 
        cumulative_revenue_lost = 0
        for month in range(1, (df['term_left'].max()+1)):
            df = df[df['term_left']>0] 
            cumulative_revenue_lost += df['instalment'].sum()
            revenue_lost.append(cumulative_revenue_lost) 
            df['term_left'] = df['term_left'] - 1 
        
        return revenue_lost

