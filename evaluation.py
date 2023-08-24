# initial imports for functions and otherwise
import pandas as pd
from prepare import prep_telco
from acquire import get_connection, get_telco_data
from modeling import dt_comp, rf_comp, knn_comp, lr_comp, model_comp
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

telco = get_telco_data()

telco = prep_telco(telco)

telco['target'] = (telco['churn'] == 'Yes')
telco = telco.drop(columns='churn')

train_validate, test = train_test_split(telco, test_size=0.2, random_state=117, stratify=telco['target'])
train, validate = train_test_split(train_validate, test_size=0.3, random_state=117, stratify=train_validate['target'])


# Isolating the key drivers
def cr_comp():
    '''
    cr_comp will take the three drivers 'contract', 'has_internet_service', and 'monthly_charges' and compare customers with and without those drivers based on the mean of the churn rate and return the percentages for us to compare.
    '''
    return print(f"\nChurn rate of customers with 2 year contract:\n{round(train[train.contract == 2].target.mean()*100, 2)}%\n\
    Churn rate of customers with 1 year contract:\n{round(train[train.contract == 1].target.mean()*100, 2)}%\n\
    Churn rate of customers with monthly contract:\n{round(train[train.contract == 0].target.mean()*100, 2)}%\n\n\
    Churn rate of customers with internet service:\n{round(train[train.has_internet_service  == 1].target.mean()*100, 2)}%\n\
    Churn rate of customers without internet service:\n{round(train[train.has_internet_service  == 0].target.mean()*100, 2)}%\n\n\
    Churn rate of customers with higher than average monthly charge:\n{round(train[train.monthly_charges > (train.monthly_charges.mean())].target.mean()*100, 2)}%\n\
    Churn rate of customers with lower than average monthly charge:\n{round(train[train.monthly_charges < (train.monthly_charges.mean())].target.mean()*100, 2)}%\
    ")


# visual confirming what was found
def driv_viz():
    '''
    driv_viz will take the three primary drivers found and return them in a catplot showing churn rate differences of customers with or without internet service compared to monthly charges seperated by contract type.
    '''
    return sns.catplot(data=train, x='has_internet_service' , y='monthly_charges', hue='target', col='contract', kind='bar')


# establishing alpha for all stats tests
α = 0.05


# defining stats testing functions
def chi_sqr_mon():
    '''
    chi_sqr_mon will perform a chi squared test to determine independence of monthly charges to churn rate and return confirmation or rejection of our null hypothesis.
    '''
    H0 = "churn rate is independent of monthly charges"
    Hα = "churn rate is dependent of monthly charges"
    observed = pd.crosstab(train['target'], train['monthly_charges'])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'P-Value: {p}\nΑlpha: {α}\n')
    if p < α:
        return print(f'We reject the null hypothesis that {H0}.\nEvidence suggests that {Hα}.')
    else:
        return print(f'We fail to reject the null hypothesis. We find insufficient evidence to support the claim that {Hα}.')


def chi_sqr_int():
    '''
    chi_sqr_mon will perform a chi squared test to determine independence of having internet service to churn rate and return confirmation or rejection of our null hypothesis.
    '''
    H0 = "churn rate is independent of if a customer has internet service"
    Hα = "churn rate is dependent on if a customer has internet service"
    observed = pd.crosstab(train['target'], train['has_internet_service'])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'P-Value: {p}\nΑlpha: {α}\n')
    if p < α:
        return print(f'We reject the null hypothesis that {H0}.\nEvidence suggests that {Hα}.')
    else:
        return print(f'We fail to reject the null hypothesis. We find insufficient evidence to support the claim that {Hα}.')
    
    
def chi_sqr_con():
    '''
    chi_sqr_mon will perform a chi squared test to determine independence of contract type to churn rate and return confirmation or rejection of our null hypothesis.
    '''
    H0 = "churn rate is independent of their contract"
    Hα = "churn rate is dependent on their contract"
    observed = pd.crosstab(train['target'], train['contract'])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'P-Value: {p}\nΑlpha: {α}\n')
    if p < α:
        return print(f'We reject the null hypothesis that {H0}.\nEvidence suggests that {Hα}.')
    else:
        return print(f'We fail to reject the null hypothesis. We find insufficient evidence to support the claim that {Hα}.')