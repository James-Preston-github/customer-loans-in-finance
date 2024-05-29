# Exploratory Data Analysis: Customer Loans in Finance

This project is part of my AI Core Data Analysis bootcamp with the aim to conduct Exploratory Data Analysis (EDA) on tabular data related to customer loan payments. 

The primary objectives include extracting data from an AWS Relational Database, transferring it to a pandas dataframe, and saving it as a CSV file for subsequent processing and analysis.The extracted data undergoes a series of transformations, including imputation and null removal, skewness optimization, outlier elimination, and correlation identification. Subsequently, analysis and visualization techniques are applied to gain valuable insights into the present condition of loans, assess current and potential losses, and identify indicators of risk.

# Usage/Installation instructions

First ensure the appropriate conda environment is set up.
Run the 'db_utils.py' file to extract the data from an AWS Relational Database and write it into the appropriate csv file. This requires the .yaml credentials for the AWS RDS.
Since this is confidential, SKIP THIS STEP, This file has already been run and the csv file has been included within this repository, as 'loan_payments.csv'.
Open and run the 'exploratory_data_analysis.ipynb' notebook. This contains the exploratory data analysis where the data is transformed to remove and impute nulls, optimise skewness, remove outliers and identify correlation.
Read through this notebook to understand the EDA process.

The 'analysis_and_visualisation.ipynb' notebook should then be run. This provides insights, conclusions and visualisations from the transformed data. Analysis on the current state of loans, current and potential losses as well as identifying risk indicating variables are provided in this notebook.

# File structure of the project

#### Analysis notebooks and files:
- **db_utils.py**: This is a python script that extracts the data from an AWS RDS using .yaml credentials that are not provided due to confidentiality. This file has already been run and the subsequent .csv file ('*loan_payments.csv*') has been included in this repository.
- **exploratory_data_analysis.ipynb**: This is the notebook in which the exploratory data analysis is conducted, this should be run and read to understand the EDA and dataframe transformation process.
- **analysis_and_visualisation.ipynb**: This is the notebook that contains analysis and visualisations of the transformed dataframe. This interactive notebook contains insights on and conclusions from the data.

#### Data:
- **loan_payments.csv**: Raw data as extracted from the cloud in *db_utils.py*.
- **loan_payments_cleaned.csv**: Cleaned data after applying transformations in *exploratory_data_analysis.ipynb* notebook.

#### Python files containing classes and methods used for analysis:
- **data_transform.py**: This is a python script which defines the DataTransform() class which is used to transform the format of the dataframe. This is imported as a module into the '*exploratory_data_analysis.ipynb*' notebook.
- **data_frame_info.py**: This is a python script that defines the DataFrameInfo() class which is used to retrive information and insights from the dataframe. This is imported as a module into the '*exploratory_data_analysis.ipynb*' notebook.
- **data_frame_transform.py**: This is a python script which defines the DataFrameTransformation() class which is used to conduct transformations on the dataframe. This is imported as a module into the '*exploratory_data_analysis.ipynb*' notebook.
- **data_plot.py**: This is a python script that defines the Plotter() class, this class is used to provide visualisations on the dataframe. This is imported as a module into the '*exploratory_data_analysis.ipynb*' notebook.

