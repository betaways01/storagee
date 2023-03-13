# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:50:30 2023

@author: agentimis1
"""
# %% Libraries for data manipulation
# %% Libraries for data manipulation
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# data pre-processing libraries
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Libraries for models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# Performance Metrics and Plotting
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# tuning
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
# %% Load data
# npath="G:/.shortcut-targets-by-id/15noBbueCApEzl_pG429IH5b85FWoTT1V/Dissertation_Analysis"
rtd = pd.read_csv('Cleaned_Data.csv', index_col=0)

# %% Constants
n = 1  # Number of interations
n_splits = 10  # Number of slipts in the cross validation

# %% Select specific colimns and re-arrange
rtd_1 = rtd.iloc[:, 5:len(rtd.axes[1])]
cols = list(rtd_1)
cols.insert(0, cols.pop(cols.index('STEMDegreeCompletion')))
rtd_1 = rtd_1.loc[:, cols]
# %% Convert to numberic, remove NAS
cols = rtd_1.columns
rtd_1[cols] = rtd_1[cols].apply(pd.to_numeric, errors='coerce')
rtd_1 = rtd_1.replace(r'^s*$', float('NaN'), regex=True)
rtd_1.dropna(inplace=True)

# Instanciate StandardScaler
scaler = MinMaxScaler()

# Select the numerical features to be scaled
num_cols = ['HSGPA',
            'ACTMath',
            'ACTEnglish',
            'EFC',
            'FamilyIncome',
            'FirstSemGPA',
            'FirstYearGPA',
            'SecondYearGPA',
            'FirstYrEarnedCreditHours',
            'SecondYrEarnedCreditHours']

# Scale the numerical features using StandardScaler
rtd_1[num_cols] = scaler.fit_transform(rtd_1[num_cols])

# %% Split The Input and output values ==============================
X = rtd_1.iloc[:, 1:len(rtd_1)].values
y = rtd_1.iloc[:, 0].values.flatten()
type(X)
type(y)

# %%
# split the data into train and test
# 70% - 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2023)

# %%
kbest = SelectKBest(score_func=chi2, k=20)  # select top 20 features
X_train_kbest = kbest.fit_transform(X_train, y_train)
X_test_kbest = kbest.transform(X_test)

# Get the selected features
selected_features = X.columns[kbest.get_support()]
print('The Selected Features are: ', selected_features)

# %%
# Split The Input and output values
X_logit = rtd_1.drop(['STEMDegreeCompletion'], axis=1)
y_logit = rtd_1['STEMDegreeCompletion']

# split the data into train and test
# 70& - 30%
X_train_logit, X_test_logit, y_train_logit, y_test_logit = train_test_split(
    X_logit, y_logit, test_size=0.3, random_state=2023)

# %%==== Create empty list to store the Mean Square Error
lr_errors = []
rf_errors = []
gmb_errors = []
ada_errros = []
svm_errors = []
dtree_errors = []
kf = StratifiedKFold(n_splits=n_splits)
# %% Predictions
for i in range(n):  # times of running each model
    for train_index, test_index in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        # Training Logistic Regression
        lr = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs')
        lr.fit(X_train, y_train)
        lr_y_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_y_pred)
        lr_errors.append(lr_acc)

        # Training Random Forest
        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                    max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=2,
                                    oob_score=False, random_state=None, verbose=0,
                                    warm_start=False)
        rf.fit(X_train, y_train)
        rf_y_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_y_pred)
        rf_errors.append(rf_acc)

        # Training GBM classifier
        gmb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bytree=0.6, gamma=5, learning_rate=0.02, max_delta_step=0,
                            max_depth=5, min_child_weight=5, missing=None, n_estimators=600,
                            n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                            silent=True, subsample=1.0)
        gmb.fit(X_train, y_train)
        gmb_y_pred = gmb.predict(X_test)
        gmb_acc = accuracy_score(y_test, gmb_y_pred)
        gmb_errors.append(gmb_acc)

        # Training ADA classifier
        ada = AdaBoostClassifier(base_estimator=None, learning_rate=0.01,
                                 n_estimators=1000,
                                 algorithm="SAMME.R", random_state=42)
        ada.fit(X_train, y_train)
        ada_y_pred = ada.predict(X_test)
        ada_acc = accuracy_score(y_test, ada_y_pred)
        ada_errros.append(ada_acc)

        # Training SVM classifier
        svm = SVC(gamma=0.1, C=10, random_state=None, )
        svm.fit(X_train, y_train)
        svm_y_pred = svm.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_y_pred)
        svm_errors.append(svm_acc)

        # Training dtree classifier
        dtree = DecisionTreeClassifier(class_weight='balanced', criterion='gini',
                                       max_depth=None, max_features=None, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0,
                                       random_state=None, splitter='best')
        dtree.fit(X_train, y_train)
        dtree_y_pred = dtree.predict(X_test)
        dtree_acc = accuracy_score(y_test, dtree_y_pred)
        dtree_errors.append(dtree_acc)

        print("Working on Iteration"+i)
# %%
Result_acc = pd.DataFrame({'Logistic Regression': lr_errors,
                           'Random_Forest': rf_errors,
                          'gmb_errors': gmb_errors,
                           'ada_errros': ada_errros,
                           'svm_errors': svm_errors,
                           'dtree_errors': dtree_errors,
                           })

# %% Write the results to a csv file
Result_acc.to_csv('/Results/Accuracies.csv')
# %% Graphs

Results_long = pd.melt(Result_acc, var_name="Method",
                       value_name="MSE")  # Wragle for plot
Crossvalidation_boxplot = sns.boxplot(data=Results_long, x="Method", y='MSE',
                                      showmeans=True, width=0.95).set(title='MSE by models', xlabel='Method', ylabel='MSE')
