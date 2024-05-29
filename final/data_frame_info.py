import pandas as pd



class DataFrameInfo:
    """    
    This class contains methods that generate useful information about a pandas DataFrame.

    Methods:
        check_dtypes(self, dataframe)
        shape(self, dataframe, print = None)
        column_list(self, dataframe, print = None)
        column_info_dataframe(self, dataframe, column_name = None)
        filter_column_info_dataframe(self, dataframe, column_to_filter, vaules_to_filter)
        get_null_percentage(self, dataframe)
        numeric_columns(self, dataframe)
        get_skewed_columns(self, dataframe, col_list, threshold) 
    """

    def check_dtypes(self, dataframe):
        '''
        Describes all columns in the DataFrame to check their data types

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.

        Returns:
            None: Prints information about a DataFrame including the index dtype and columns, non-null values and memory usage
        '''
        return dataframe.info()
 
    def shape(self, dataframe, print = None):
        '''
        Prints out the shape of the DataFrame 

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.
            print (str): Determines whether the method prints a statement or not. Default is None.

        Returns:
            dataframe.shape: Number of columns and rows as a tuple. If print parameter is not None, will print the statement too.
        '''
        if print == None:
            return dataframe.shape
        else:
            print(f'The DataFrame has {dataframe.shape[1]} columns and {dataframe.shape[0]} rows.')

    def column_list(self, dataframe, print = None):
        '''
        Gets list of columns in a pandas dataframe

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.
            print (str): Determines whether the method prints a statement or not. Default is None.

        Returns:
            col_list (list): List of columns in the dataframe. If print parameter is not None, will print the statement too.
        '''
        col_list = dataframe.columns.to_list()
        if print == None:
            return col_list
        else:
            print(f'The DataFrame has column names: {col_list}.')
           
    def column_info_dataframe(self, dataframe, column_name = None):
        '''
        Counts distinct values in categorical columns of a pandas dataframe. 
        If no column is specified returns a dataframe of all column information

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.
            column_name (str): specific column to apply this method to. Default is None meaning all columns will be processed.

        Returns:
            list_dataframe (pd.DataFrame): contains list of distinct values, the count and datatype of each selected column.
        '''

        col_list = dataframe.columns.to_list()
        list = [(x, dataframe[x].unique(), len(dataframe[x].unique()), dataframe[x].dtype, dataframe[x].isna().sum()) for x in col_list]
        list_dataframe = pd.DataFrame(list, columns = ['Column Name', 'Distinct Values', 'Distinct Values Count', 'Data Type', 'Null Values Count'], index = col_list)
        if column_name == None:
            return list_dataframe    
        else:
            return list_dataframe.loc[column_name]
    
    def filter_column_info_dataframe(self, dataframe, column_to_filter, vaules_to_filter):
        '''
        Filters the column_info_dataframe() method by a column and a list of values.

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.
            column_to_filter (str): Column to fiter.
            vaules_to_filter (list): list of value(s) to filter the column by.

        Returns:
            filtered_df (pd.DataFrame): contains list of distinct values, the count and datatype of each selected column. Filtered as desired.
        '''
        column_info = self.column_info_dataframe(dataframe)
        filtered_df = column_info.loc[column_info[column_to_filter].isin(vaules_to_filter)]
        return filtered_df

    def get_null_percentage(self, dataframe):
        '''
        Generates a count/percentage count of NULL values in each column.

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.

        Returns:
            null_df (pd.DataFrame): A dataframe of column names and their percentage of null values.
        '''        
        
        null_percentage_dict = {}
        for col in dataframe.columns:
            percentage = dataframe[col].isnull().sum()/len(dataframe)*100
            null_percentage_dict[col] = round(percentage, 2) #for each key=col, value=rounded percentage
        null_percentage_dict = sorted(null_percentage_dict.items(), key=lambda x:x[1], reverse=True) # sort with highest %. But returns a list of tuples
        null_df = pd.DataFrame(null_percentage_dict, columns=['Column Name', 'Null Values %']) #display as dataframe for neatness
        return null_df    

    def numeric_columns(self, dataframe):
        '''
        Gets a list of columns in a dataframe that are numeric (int or float data type)

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.

        Returns:
            numeric_cols (list): List of columns in the dataframe that are numeric.
        '''

        numeric_cols = []
        for col in dataframe.columns:
            #print(df[col].dtype)
            if dataframe[col].dtype in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
            #    print(col)
                numeric_cols.append(col)
        return numeric_cols
    
    def get_skewed_columns(self, dataframe, col_list, threshold):

        '''
        Gets list of columns in a dataframe that are skewed as defined by a specified threshold. 
        Will only work for columns that are of a numeric datatype.

        Parameters:
            dataframe (pd.DataFrame): The dataframe to which this method will be applied.
            col_list (list): List of columns to which this method will be applied/
            threshold (float): threshold defining level of skew

        Returns:
            skewed_cols (list): List of skewed columns
        '''
        skewed_cols = {}
        for col in col_list:
            skew = dataframe[col].skew()
            if abs(skew) > threshold:
                skewed_cols[col] = round(skew, 2)
        return skewed_cols
    

 