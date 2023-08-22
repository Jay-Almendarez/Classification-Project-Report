# initial imports for functions and otherwise
import pandas as pd
from prepare import prep_telco
from acquire import get_connection, get_telco_data
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt


telco = get_telco_data()
telco = prep_telco(telco)
telco['target'] = (telco['churn'] == 'Yes')
telco = telco.drop(columns='churn')


train_validate, test = train_test_split(telco, test_size=0.2, random_state=117, stratify=telco['target'])
train, validate = train_test_split(train_validate, test_size=0.3, random_state=117, stratify=train_validate['target'])


X_train1 = train.drop(columns=['target'])
y_train1 = train['target']

X_validate1 = validate.drop(columns=['target'])
y_validate1 = validate['target']

X_test1 = test.drop(columns=['target'])
y_test1 = test['target']

telco['baseline'] = telco['target'].value_counts().idxmax()
baseline_accuracy = (telco.baseline == telco.target).mean()


def dt_comp():
    '''
    dt_comp will determine in a range of 1 to 30, what the best max depth for a decision tree model is by graphing the training and validate set side by side for visual comparison.
    '''
    k_range = range(1,30)
    train_score = []
    validate_score = []
    for k in k_range:
        clf = DecisionTreeClassifier(max_depth=k, random_state=117)
        clf.fit(X_train1, y_train1)
        train_score.append(clf.score(X_train1, y_train1))
        validate_score.append(clf.score(X_validate1, y_validate1))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_score, label = 'Train')
    plt.plot(k_range, validate_score, label = 'Validate')
    plt.legend()
    return plt.show()


def rf_comp():
    '''
    rf_comp will determine in a range of 1 to 30, what the best max depth for a random forest model is by graphing the training and validate set side by side for visual comparison.
    '''
    k_range = range(1,30)
    train_score = []
    validate_score = []
    for k in k_range:
        rf = RandomForestClassifier(max_depth = k, random_state=117)
        rf.fit(X_train1, y_train1)
        train_score.append(rf.score(X_train1, y_train1))
        validate_score.append(rf.score(X_validate1, y_validate1))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_score, label = 'Train')
    plt.plot(k_range, validate_score, label = 'Validate')
    plt.legend()
    return plt.show()


def knn_comp():
    '''
    knn_comp will determine in a range of 1 to 30, what the optimal amount of n_neighbors is for a knn model by graphing the training and validate set side by side for visual comparison.
    '''
    k_range = range(1,30)
    train_score = []
    validate_score = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train1, y_train1)
        train_score.append(knn.score(X_train1, y_train1))
        validate_score.append(knn.score(X_validate1, y_validate1))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_score, label = 'Train')
    plt.plot(k_range, validate_score, label = 'Validate')
    plt.legend()
    return plt.show()


def lr_comp():
    '''
    lr_comp will generate 5 logistic regression models with varying feature selection and return the accuracy percentages compared to the baseline accuracy.
    '''
    X_train1 = train.drop(columns=['target'])
    y_train1 = train['target']

    X_validate1 = validate.drop(columns=['target'])
    y_validate1 = validate['target']

    X_test1 = test.drop(columns=['target'])
    y_test1 = test['target']


    lr1 = LogisticRegression(random_state=117)

    lr1.fit(X_train1, y_train1)

    lr_tr_acc1 = lr1.score(X_train1,y_train1)


    lr1.fit(X_validate1, y_validate1)

    lr_val_acc1 = lr1.score(X_validate1, y_validate1)



    X_train2 = train.drop(columns=['target', 'senior_citizen', 'has_multiple_lines'])
    y_train2 = train['target']

    X_validate2 = validate.drop(columns=['target', 'senior_citizen', 'has_multiple_lines'])
    y_validate2 = validate['target']

    X_test2 = test.drop(columns=['target', 'senior_citizen', 'has_multiple_lines'])
    y_test2 = test['target']


    lr2 = LogisticRegression()

    lr2.fit(X_train2, y_train2)

    lr_tr_acc2 = lr2.score(X_train2,y_train2)


    lr2.fit(X_validate2, y_validate2)

    lr_val_acc2 = lr2.score(X_validate2, y_validate2)



    X_train3 = train.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_'])
    y_train3 = train['target']

    X_validate3 = validate.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_'])
    y_validate3 = validate['target']

    X_test3 = test.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_'])
    y_test3 = test['target']


    lr3 = LogisticRegression()

    lr3.fit(X_train3, y_train3)

    lr_tr_acc3 = lr3.score(X_train3,y_train3)


    lr3.fit(X_validate3, y_validate3)

    lr_val_acc3 = lr3.score(X_validate3, y_validate3)



    X_train4 = train.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_', 'has_dependents', 'has_phone_service',
                                   'has_partner'])
    y_train4 = train['target']

    X_validate4 = validate.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_', 'has_dependents', 'has_phone_service',
                                         'has_partner'])
    y_validate4 = validate['target']

    X_test4 = test.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_', 'has_dependents', 'has_phone_service',
                                 'has_partner'])
    y_test4 = test['target']


    lr4 = LogisticRegression()

    lr4.fit(X_train4, y_train4)

    lr_tr_acc4 = lr4.score(X_train4,y_train4)


    lr4.fit(X_validate4, y_validate4)

    lr_val_acc4 = lr4.score(X_validate4, y_validate4)



    X_train5 = train.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_', 'has_dependents', 'has_phone_service',
                                   'has_partner', 'has_amenities', 'has_internet_service', 'contract'])
    y_train5 = train['target']

    X_validate5 = validate.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_', 'has_dependents', 'has_phone_service',
                                   'has_partner', 'has_amenities', 'has_internet_service', 'contract'])
    y_validate5 = validate['target']

    X_test5 = test.drop(columns=['target', 'senior_citizen', 'has_multiple_lines', 'internet_service', 'has_automatic_payment', 'gender_', 'has_dependents', 'has_phone_service',
                                   'has_partner', 'has_amenities', 'has_internet_service', 'contract'])
    y_test5 = test['target']


    lr5 = LogisticRegression()

    lr5.fit(X_train5, y_train5)

    lr_tr_acc5 = lr5.score(X_train5,y_train5)


    lr5.fit(X_validate5, y_validate5)

    lr_val_acc5 = lr5.score(X_validate5, y_validate5)


    return print(f'Logistic Regression Model Accuracy Scores:\n\n\
Baseline Accuracy Score:\n{baseline_accuracy:2%}\n\n\
Model 1 Train Accuracy Score:\n{lr_tr_acc1:2%}\n\
Model 1 Validate Accuracy Score:\n{lr_val_acc1:2%}\n\n\
Model 2 Train Accuracy Score:\n{lr_tr_acc2:2%}\n\
Model 2 Validate Accuracy Score:\n{lr_val_acc2:2%}\n\n\
Model 3 Train Accuracy Score:\n{lr_tr_acc3:2%}\n\
Model 3 Validate Accuracy Score:\n{lr_val_acc3:2%}\n\n\
Model 4 Train Accuracy Score:\n{lr_tr_acc4:2%}\n\
Model 4 Validate Accuracy Score:\n{lr_val_acc4:2%}\n\n\
Model 5 Train Accuracy Score:\n{lr_tr_acc5:2%}\n\
Model 5 Validate Accuracy Score:\n{lr_val_acc5:2%}\
    ')


def model_comp():
    '''
    model_comp will take the best classification models created and print the training and validation set alongside the baseline accuracy to help determine the best model for use.
    '''
    # Best KNN Model
    knn1 = KNeighborsClassifier(n_neighbors=15)

    knn1.fit(X_train1, y_train1)

    knn_tr_acc = knn1.score(X_train1,y_train1)


    knn1.fit(X_validate1, y_validate1)

    knn_val_acc = knn1.score(X_validate1, y_validate1)


    # Best Random Forest Model
    rf1 = RandomForestClassifier(max_depth=5)

    rf1.fit(X_train1, y_train1)

    rf_tr_acc = rf1.score(X_train1, y_train1)


    rf1.fit(X_validate1, y_validate1)

    rf_val_acc = rf1.score(X_validate1, y_validate1)



    # Best Decision Tree Model
    clf1 = DecisionTreeClassifier(max_depth=5)

    clf1.fit(X_train1, y_train1)

    dt_tr_acc = clf1.score(X_train1, y_train1)


    clf1.fit(X_validate1, y_validate1)

    dt_val_acc = clf1.score(X_validate1, y_validate1)
    
    
    
    # Best Logistic Regression Model
    lr1 = LogisticRegression()

    lr1.fit(X_train1, y_train1)

    lr_tr_acc1 = lr1.score(X_train1,y_train1)


    lr1.fit(X_validate1, y_validate1)

    lr_val_acc1 = lr1.score(X_validate1, y_validate1)
    
    return print(f'Model Train and Validate Accuracy Scores: \n\n\
Baseline Accuracy: \n{baseline_accuracy:2%}\n\n\
Decision Tree Train Score: \n{dt_tr_acc:2%}\n\
Decision Tree Validate Score: \n{dt_val_acc:2%}\n\n\
Random Forest Train Score: \n{rf_tr_acc:2%}\n\
Random Forest Validate Score: \n{rf_val_acc:2%}\n\n\
K Nearest Neighbor Train Score: \n{knn_tr_acc:2%}\n\
K Nearest Neighbor Validate Score: \n{knn_val_acc:2%}\n\n\
Logistic Regression Train Score: \n{lr_tr_acc1:2%}\n\
Logistic Regression Validate Score: \n{lr_val_acc1:2%}\n\n\
')