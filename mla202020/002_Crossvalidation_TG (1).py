# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:50:30 2023

@author: agentimis1
"""
#%% Libraries for data manipulation
import os 
import pandas as pd
#import numpy as np
from sklearn.model_selection import StratifiedKFold
#%% Libraries for models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#%% Library for Metrics and Plotting
from sklearn.metrics import accuracy_score
import seaborn as sns
#import matplotlib.pyplot as plt 
#%% Load data
#npath="G:/.shortcut-targets-by-id/15noBbueCApEzl_pG429IH5b85FWoTT1V/Dissertation_Analysis"
npath=os.path.abspath(os.pardir)
rtd=pd.read_csv(npath+'/Data/Cleaned_Data.csv',index_col=0)

#%% Constants
n = 1 #Number of interations
n_splits = 10 #Number of slipts in the cross validation

#%% Select specific colimns and re-arrange
rtd_1=rtd.iloc[:,5:len(rtd.axes[1])]
cols = list(rtd_1) 
cols.insert(0, cols.pop(cols.index('STEMDegreeCompletion')))
rtd_1=rtd_1.loc[:,cols]
#%% Convert to numberic, remove NAS
cols = rtd_1.columns
rtd_1[cols] = rtd_1[cols].apply(pd.to_numeric, errors='coerce')
rtd_1 = rtd_1.replace(r'^s*$', float('NaN'), regex = True)
rtd_1.dropna(inplace=True)
#%% Split The Input and output values ============================== 
X=rtd_1.iloc[:,1:len(rtd_1)].values
y=rtd_1.iloc[:,0].values.flatten()
type(X)
type(y)
#%%==== Create empty list to store the Mean Square Error
lr_errors = []
rf_errors = []
gmb_errors=[]
ada_errros=[]
svm_errors=[]
dtree_errors=[]
kf = StratifiedKFold(n_splits=n_splits)
#%% Predictions 
for i in range(n): #times of running each model  
    for train_index, test_index in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
        # Training Logistic Regression
        lr = LogisticRegression(penalty='none')
        lr.fit(X_train, y_train)
        lr_y_pred=lr.predict(X_test)          
        lr_acc=accuracy_score(y_test, lr_y_pred)
        lr_errors.append(lr_acc)
       
        # Training Random Forest
        rf = RandomForestClassifier(n_estimators=100)  
        rf.fit(X_train, y_train)  
        rf_y_pred=rf.predict(X_test)          
        rf_acc=accuracy_score(y_test, rf_y_pred)
        rf_errors.append(rf_acc)
         
        # Training GBM classifier
        gmb = XGBClassifier()
        gmb.fit(X_train, y_train)  
        gmb_y_pred=gmb.predict(X_test)          
        gmb_acc=accuracy_score(y_test, gmb_y_pred)
        gmb_errors.append(gmb_acc)
      
        # Training ADA classifier
        ada = AdaBoostClassifier()
        ada.fit(X_train, y_train)  
        ada_y_pred=ada.predict(X_test)          
        ada_acc=accuracy_score(y_test, ada_y_pred)
        ada_errros.append(ada_acc)
        
        # Training SVM classifier
        svm = SVC()
        svm.fit(X_train, y_train)  
        svm_y_pred=svm.predict(X_test)          
        svm_acc=accuracy_score(y_test, svm_y_pred)
        svm_errors.append(svm_acc)
      
        # Training dtree classifier
        dtree = DecisionTreeClassifier(class_weight = 'balanced', criterion='gini',
                               max_depth=None, max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, 
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, 
                               random_state=None, splitter='best')
        dtree.fit(X_train, y_train)  
        dtree_y_pred=dtree.predict(X_test)          
        dtree_acc=accuracy_score(y_test, dtree_y_pred)
        dtree_errors.append(dtree_acc)
        
        print("Working on Iteration"+i)
#%%
Result_acc=pd.DataFrame({'Logistic Regression': lr_errors,
                         'Random_Forest':rf_errors,
                        'gmb_errors':gmb_errors,
                        'ada_errros':ada_errros,
                        'svm_errors':svm_errors,
                        'dtree_errors':dtree_errors,
                        })

#%% Write the results to a csv file
Result_acc.to_csv(npath+'/Results/Accuracies.csv')
#%% Graphs

Results_long = pd.melt(Result_acc, var_name="Method", value_name="MSE") #Wragle for plot
Crossvalidation_boxplot=sns.boxplot(data=Results_long, x="Method",y='MSE', 
                           showmeans = True, width=0.95).set(title='MSE by models', xlabel='Method', ylabel='MSE')


