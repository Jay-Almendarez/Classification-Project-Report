import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pydataset import data
from scipy import stats
import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    get_connection will determine the database we are wanting to access, and load the database along with env stored values like username, password, and host
    to create the url needed for SQL to read the correct database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    '''
    get_titanic_data will determine if 'titanic.csv' exists, if it does, it will load the dataframe titanic_db,
    if it does not exist, it will write the dataframe titanic_db into a .csv
    '''
    file_name = 'titanic.csv'
    
    
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    else:
        query = 'SELECT * FROM passengers'
        connection = get_connection('titanic_db')
        df = pd.read_sql(query, connection)
        df.to_csv('titanic.csv', index=False)
        return df

def get_iris_data():
    '''
    get_iris_data will determine if 'iris.csv' exists, if it does, it will load the dataframe iris_db,
    if it does not exist, it will write the dataframe iris_db into a .csv
    '''
    file_name = 'iris.csv'
    
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    else:
        query = 'SELECT * FROM measurements JOIN species USING (species_id)'
        connection = get_connection('iris_db')
        df = pd.read_sql(query, connection)
        df.to_csv('iris.csv', index=False)
        return df

def get_telco_data():
    '''
    get_telco_data will determine if 'telco.csv' exists, if it does, it will load the dataframe telco_churn,
    if it does not exist, it will write the dataframe telco_churn into a .csv
    '''
    file_name = 'telco.csv'
    
    
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    else:
        query = 'SELECT * FROM customers LEFT JOIN contract_types USING (contract_type_id) LEFT JOIN internet_service_types USING (internet_service_type_id) LEFT JOIN payment_types USING (payment_type_id)'
        connection = get_connection('telco_churn')
        df = pd.read_sql(query, connection)
        df.to_csv('telco.csv', index=False)
        return df