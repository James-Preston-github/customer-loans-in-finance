import pandas as pd
import psycopg2
import yaml




yaml_file_path = 'credentials.yaml'

def yaml_reader():
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f) 
    return data
full_dataset = yaml_reader()


class RDSDatabaseConnector:
    def __init__(self, dictionary):
        self.dictionary = dictionary


    def extract_dataframe(self):
        with psycopg2.connect(host=full_dataset['RDS_HOST'], user=full_dataset['RDS_USER'], password=full_dataset['RDS_PASSWORD'], dbname=full_dataset['RDS_DATABASE'], port=full_dataset['RDS_PORT']) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM loan_payments")
            columns = [desc[0] for desc in cur.description]
            results_df = pd.DataFrame(cur.fetchall(), columns=columns)
            cur.close()
            conn.close()
            return results_df
            
if __name__ == '__main__':
    class_instance = RDSDatabaseConnector(full_dataset)
    loan_payments = class_instance.extract_dataframe()
    loan_payments.to_csv('loan_payments.csv')
    print(loan_payments.head())
