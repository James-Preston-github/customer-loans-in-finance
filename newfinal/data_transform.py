import pandas as pd

class DataTransform:
    """    
    Class contains methods convert columns to the correct format
    
    Methods:
        convert_to_category(self, dataframe, column_name)
        convert_to_timeperiod(self, dataframe, column_name, frequency='M')
        convert_to_integer(self, dataframe, column_name)
        convert_to_string(self, dataframe, column_name)
    """
    
    def convert_to_category(self, dataframe, column_name):
        """
        Converts a column in a pandas DataFrame to category data type.

        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
            column_name (str): The column of the DataFrame to perform the conversion on.

        Returns:
            DataFrame (pd.DataFrame): the updated DataFrame.
        """

        dataframe[column_name] = dataframe[column_name].astype('category')
        return dataframe

    def convert_to_timeperiod(self, dataframe, column_name, frequency='M'):
        """
        Converts a column in a pandas DataFrame to the Period data type.

        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
            column_name (str): The column of the DataFrame to perform the conversion on.
            frequency (str): The frequency of the Period (default is 'M' for monthly)

        Returns:
            DataFrame (pd.DataFrame): the updated DataFrame.
        """
        # Convert the specified column to datetime
        dataframe[column_name] = pd.to_datetime(dataframe[column_name], errors='coerce')

        # Convert datetime to Period with the specified frequency
        dataframe[column_name] = dataframe[column_name].dt.to_period(frequency)
        return dataframe
    
    def convert_to_integer(self, dataframe, column_name):
        """
        Converts a column in a pandas DataFrame to integer data type, ignoring null values.

        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
            column_name (str): The column of the DataFrame to perform the conversion on.

        Returns:
            DataFrame (pd.DataFrame): the updated DataFrame.
        """
        # Use pd.to_numeric with errors='coerce' to handle null values
        dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce', downcast='integer')
        return dataframe

    def convert_to_string(self, dataframe, column_name):
        """
        Converts a column in a pandas DataFrame to string data type.

        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
            column_name (str): The column of the DataFrame to perform the conversion on.

        Returns:
            DataFrame (pd.DataFrame): the updated DataFrame.
        """
        dataframe[column_name] = dataframe[column_name].astype(str)
        return dataframe

