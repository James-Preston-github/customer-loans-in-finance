from matplotlib import pyplot
from scipy import stats
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


class Plotter:

    '''
    This class is used to plot visualisations of the data.

    Methods:
        missing_matrix(self, DataFrame)
        compare_skewness_transformations(self, DataFrame, column_name)
        facet_grid_box_plot(self, DataFrame, column_names)
        correlation_matrix(self, DataFrame)
        bar_chart(self, independant_categories, dependant_variables, title=None, y_label=None, x_label=None)
        pie_chart(self, labels, values, title=None)
    '''

    def missing_matrix(self, DataFrame):

        '''
        This method plots a matrix displaying missing or null data points within the DataFrame.
        
        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: A matrix plot showing all the missing or null data points in each column in white.
        '''

        return msno.matrix(DataFrame)
    

    def compare_skewness_transformations(self, DataFrame, column_name):
        '''
        This method compares the result of yeo-johnson and log transformations to correct skewness.
        
        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
            column_name (str): The column of the DataFrame to perform the transformations on

        Returns:
            3 plots showing histograms for orginal data, log transformed, and yeo-johnson transformed. As well as the skew printed on each figure
        '''

        transformed_df = DataFrame.copy()
        transformed_df['yeo-johnson'] = pd.Series(stats.yeojohnson(DataFrame[column_name])[0]).values # Perform yeo-johnson transformation and add values as new column in DataFrame copy.
        transformed_df["log_transformed"] = DataFrame[column_name].map(lambda x: np.log(x) if x > 0 else 0) # Log transformation applied to value in column, if value is 0 then no transformation is done and added to new column in DataFrame copy.
            
        fig, axes = pyplot.subplots(nrows=1, ncols=3, figsize=(16, 4)) # Create a 2x3 grid
        pyplot.suptitle(column_name, fontsize='x-large') # Add large title for entire plot.

        axes[0].set_title('Original Histogram')
        axes[1].set_title('Log Transformed Histogram')
        axes[2].set_title('Yeo-Johnson Transformed Histogram')
        
            # Original Histogram
        sns.histplot(DataFrame[column_name], kde=True, ax=axes[0]) 
        axes[0].text(0.5, 0.95, f'Skewness: {DataFrame[column_name].skew():.2f}', ha='center', va='top', transform=axes[0].transAxes) # Add skewness
        # Log transformed Histogram
        sns.histplot(transformed_df['log_transformed'], kde=True, ax=axes[1]) 
        axes[1].text(0.5, 0.95, f'Skewness: {transformed_df["log_transformed"].skew():.2f}', ha='center', va='top', transform=axes[1].transAxes) # Add skewness
        # Yeo Johnson Histogram
        sns.histplot(transformed_df['yeo-johnson'], kde=True, ax=axes[2]) 
        axes[2].text(0.5, 0.95, f'Skewness: {transformed_df["yeo-johnson"].skew():.2f}', ha='center', va='top', transform=axes[2].transAxes) # Add skewness

        return pyplot.show()


    def facet_grid_box_plot(self, DataFrame, column_names):

        '''
        This method is used to return a Facet Grid containing box-plots for a list of columns.

        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
            column_names (list): A list of names of columns which will be plotted.

        Returns:
            facet_grid (sns.FacetGrid): A facetgrid containing the box-plots of each of the variables.
        '''

        melted_df = pd.melt(DataFrame, value_vars=column_names) # Melt the DataFrame to reshape it.
        facet_grid = sns.FacetGrid(melted_df, col="variable",  col_wrap=3, sharex=False, sharey=False) # Create the facet grid
        facet_grid = facet_grid.map(sns.boxplot, "value", flierprops=dict(marker='x', markeredgecolor='red')) # Map box-plot onto each plot on grid.
        return facet_grid
    

    def correlation_matrix(self, DataFrame):
        """
        Calculate the correlation matrix for columns in a pandas DataFrame.

        Parameters:
            DataFrame (pd.DataFrame): The DataFrame to which this method will be applied.
        
        Returns:
            Correlation_matrix: pandas DataFrame, the correlation matrix
        """
        correlation_matrix = DataFrame.corr()
        styled_correlation_matrix = correlation_matrix.style.background_gradient(cmap='coolwarm')
        return styled_correlation_matrix
    
    def bar_chart(self, independant_categories, dependant_variables, title=None, y_label=None, x_label=None): 
        '''
        This method is used to generate a bar chart plot of categorical data.

        Parameters:
            independant_categories (list): The names of the categories in a list.
            dependant_variables (list): The respective dependant variables in a list.
            title (str): DEFAULT = None, the title of the plot.
            y_label (str): DEFAULT = None, the label for the y-axis.
            x_label (str): DEFAULT = None, the label for the x-axis.

        Returns:
            matplotlib.pyplot.figure: a bar plot of the data.
        '''
        pyplot.figure(figsize=(8, 4))
        sns.barplot(x=independant_categories, y=dependant_variables) # Generating the bar plot and setting the independant and dependant variables.
        if y_label != None: # If a 'y_label' is provided.
            pyplot.ylabel(y_label)
        if x_label != None: # If a 'x_label' is provided.
            pyplot.xlabel(x_label)
        if title != None: # If a 'title' is provided.
            pyplot.title(title)
        return pyplot.show()


    def pie_chart(self, labels, values, title=None):
        '''
        This method is used to generate a bar chart plot of categorical data.

        Parameters:
            labels (list): The names of the categories in a list.
            sizes (list): The respective dependant variables in a list.
            title (str): DEFAULT = None, the title of the plot.

        Returns:
            matplotlib.pyplot.figure: a pie chart plot of the data.
        '''
        colours = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        pyplot.pie(values, labels=labels, colors=colours, autopct='%1.1f%%', startangle=90) # Generate pie chart.
        if title != None: # If a title is provided.
            pyplot.title(title)
        return pyplot.show()