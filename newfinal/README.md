# Exploratory Data Analysis: Customer Loans in Finance

This is a project to extract, clean, and analyse data that I have done as part of a coding training programme.

The main tasks are to extract data from an AWS Relational Database, load it into a pandas dataframe, and save it as a CSV file. The data is then transformed through imputation, null value removal, skewness adjustment, outlier removal, and correlation identification. Finally, analysis and visualization techniques are used to understand the state of loans, assess losses, and identify risk indicators.

## Installation instructions

Most of the downloading of the data happened already so the data is stored in the csv files (in different forms of cleanliness), you can therefore just run the ipynb as it pulls from there but if you want to see how it would work look at db_utils.py and this will show the steps I took. So look at the ipynbs to see the actual product of the project

# File structure of the project

#### Analysis notebooks and files:
- **db_utils.py**: This is a python script that extracts the data from an AWS RDS using .yaml credentials that are not provided due to confidentiality. This file has already been run and the subsequent .csv file ('*loan_payments.csv*') has been included in this repository.
- **exploratory_data_analysis.ipynb**: This is the notebook in which the exploratory data analysis is conducted, this should be run and read to understand the EDA and dataframe transformation process.
- **analysis_and_visualisation.ipynb**: This is the notebook that contains analysis and visualisations of the transformed dataframe. This interactive notebook contains insights on and conclusions from the data.

