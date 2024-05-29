#new class called RDSDatabaseConnector. This class will contain the methods which you will use to extract data from the RDS database.
import yaml
import pandas as pd
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    
    # Step 4: Write the __init__ method of your RDSDatabaseConnector class. 
    # It should take in as a parameter a dictionary of credentials which your function from the previous step will extract.
    def __init__(self, credentials_dict):
        self.credentials_dict = credentials_dict
    
    # Step 5: Define a method in your RDSDatabaseConnector class which initialises a SQLAlchemy engine from the credentials provided to your class. 
    # This engine object together with the Pandas library will allow you to extract data from the database.
    def create_engine(self):
        self.engine = create_engine(f"postgresql+psycopg2://{self.credentials_dict['RDS_USER']}:{self.credentials_dict['RDS_PASSWORD']}@{self.credentials_dict['RDS_HOST']}:{self.credentials_dict['RDS_PORT']}/{self.credentials_dict['RDS_DATABASE']}")
    
    #Step 6: Develop a method which extracts data from the RDS database and returns it as a Pandas DataFrame. 
    # The data is stored in a table called loan_payments.
    def extract_loans(self):
        with self.engine.connect() as connection:
            self.loan_payments_df = pd.read_sql_table('loan_payments', self.engine)
            return self.loan_payments_df



# function which loads the credentials.yaml file and returns the data dictionary contained within. 
# This will be be passed to your RDSDatabaseConnector as an argument which the class will use to connect to the remote database
def load_credentials():
    with open('credentials.yaml', 'r') as file:
        return yaml.safe_load(file)

credentials_dict = load_credentials()


#Step 7: Now create another function which saves the data to an appropriate file format to your local machine. 
# This should speed up loading up the data when you're performing your EDA/analysis tasks. 
# The function should save the data in .csv format, since we're dealing with tabular data.
def save_data_to_csv(loans_df):
    with open('loan_payments.csv', 'w') as file:
        loans_df.to_csv(file, encoding= 'utf-8', index= False)

#create an instance and save the data to csv
connector = RDSDatabaseConnector(credentials_dict)
connector.create_engine()
extracted_df = connector.extract_loans()
save_data_to_csv(extracted_df)

#load extracted data into a pandas df
def create_loans_dataframe():
    return pd.read_csv('loan_payments.csv')

