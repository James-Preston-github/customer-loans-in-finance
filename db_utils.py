import yaml
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import psycopg2



yaml_file_path = 'credentials.yaml'

def yaml_reader():
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f) 
    return data
data = yaml_reader()


class RDSDatabaseConnector:
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def sql_initialiser(self):    
        self.engine = create_engine(self.dictionary['RDS_HOST'])
        self.Session = sessionmaker(bind=self.engine)
    

    def extract_dataframe(self):
        with psycopg2.connect(host=data['RDS_HOST'], user=data['RDS_USER'], password=data['RDS_PASSWORD'], dbname=data['RDS_DATABASE'], port=data['RDS_PORT']) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM loan_payments")
            columns = [desc[0] for desc in cur.description]
            results_df = pd.DataFrame(cur.fetchall(), columns=columns)
            return results_df
            cur.close()
            conn.close()
            

class_instance = RDSDatabaseConnector(yaml_reader())
loan_payments = class_instance.extract_dataframe()
loan_payments.to_csv('loan_payments.csv')
