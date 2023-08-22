from sklearn.model_selection import train_test_split
from acquire import get_connection
from acquire import get_telco_data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def tvt(df, target):
    '''
    tvt will take in a DataFrame and return train, validate, and test DataFrames; stratify on whatever you decide as the target in brackets and quotations. tvt will also set a random state of 117.
    For example: tvt(df,['survived']) will return the dataframe (in this case the titanic dataframe and stratify by 'survived').
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(telco, test_size=0.2, random_state=117, stratify=telco['churn'])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=117,stratify=train_validate['churn'])
    return train, validate, test

telco = get_telco_data()

def dropped(telco,columns={'sepal_length', 'petal_width'}):
    '''
    dropped will take the dataframe and remove any columns that are indicated,
    hopefully making the process faster
    '''
    telco = telco.drop(columns=columns)
    return telco

def prep_telco(telco):
    '''
    prep_telco will do all the cleaning we need of the database 'telco_churn'.
    It will:
    - drop unnecessary columns ('payment_type_id', 'internet_service_type_id', 'contract_type_id')
    - fill in null values from columns (embarked' and 'age')
    - encode and rename the many categorical columns:
      - change 'gender' to gender_' as 0 for female, 1 for male, 
      - for most of the other boolean columns 0 signifies the lack of having, 1 is 'No', and 1 is 'Yes':
      - 'partner' to 'has_partner', 'dependents' to 'has_dependents', 'phone_service' to 'has_phone_service', 'multiple_lines' to 'has_multiple_lines',
      - 'online_security' to 'has_online_security', 'online_backup' to 'has_online_backup', 'device_protection' to 'has_device_protection', 'tech_support' to 'has_tech_support',
      - 'streaming_tv' to 'has_streaming_tv', 'streaming_movies' to 'has_streaming_movies', 'churn' to 'churned', 'contract_type' to 'contract', 'internet_service_type' to 'internet_service'
      - create dummy columns for 'payment_type'
    - concatenate the previous dataframe to the new ones with dummy variables
    
    return: concatenated and cleaned dataframe 'telco_churn' as 'df'
    '''
    telco = dropped(telco, columns={'internet_service_type_id', 'contract_type_id', 'payment_type_id'})
    telco['gender_'] = telco['gender'].map({'Female': 0, 'Male': 1})
    telco['has_partner'] = telco['partner'].map({'Yes': 1, 'No': 0})
    telco['has_dependents'] = telco['dependents'].map({'Yes': 1, 'No': 0})
    telco['has_phone_service'] = telco['phone_service'].map({'Yes': 1, 'No': 0})
    telco['has_paperless_billing'] = telco['paperless_billing'].map({'Yes': 1, 'No': 0})
    telco['has_tech_support'] = telco['tech_support'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    telco['has_online_security'] = telco['online_security'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    telco['has_online_backup'] = telco['online_backup'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    telco['has_streaming_tv'] = telco['streaming_tv'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    telco['has_streaming_movies'] = telco['streaming_movies'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    telco['has_device_protection'] = telco['device_protection'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    telco['has_multiple_lines'] = telco['multiple_lines'].map({'Yes': 2, 'No': 1, 'No phone service': 0})
    telco['contract'] = telco['contract_type'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})
    telco['internet_service_type'] = telco['internet_service_type'].fillna(value='No internet service')
    telco['internet_service'] = telco['internet_service_type'].map({'Fiber optic': 2, 'DSL': 1, 'No internet service': 0})
    telco['has_automatic_payment'] = telco['payment_type'].map({'Bank transfer (automatic)' : 1, 'Credit card (automatic)': 1, 'Electronic check': 0, 'Mailed check': 0})
    amenities = pd.DataFrame({
    'Has Tech Support': telco['has_tech_support'], 'Has Online Security': telco['has_online_security'], 'Has Paperless Billing': telco['has_paperless_billing'],
    'Has Online Backup': telco['has_online_backup'], 'Has Streaming TV': telco['has_streaming_tv'],
    'Has Streaming Movies': telco['has_streaming_movies'], 'Has Device Protection': telco['has_device_protection']})
    telco['has_amenities'] = (amenities.T.sum()>10).astype(int)
    telco['has_internet_service'] = (amenities.T.sum()>0).astype(int)

    telco = dropped(telco, columns ={'gender','partner', 'customer_id',
                                    'has_tech_support','has_online_security',
                                    'has_paperless_billing','has_streaming_movies',
                                    'has_online_backup','has_streaming_tv','has_device_protection',
                                    'dependents', 'phone_service',
                                    'paperless_billing',
                                    'multiple_lines',
                                     'online_security',
                                     'online_backup',
                                     'device_protection', 
                                     'tech_support',
                                     'streaming_tv',
                                     'streaming_movies', 
                                     'contract_type', 
                                     'internet_service_type', 'payment_type'})
    
    telco['total_charges'] = telco['total_charges'].str.replace(' ', '0').astype(float)
    return telco
