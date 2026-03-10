import pandas as pd
import numpy as np
from numba.core.types import none
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report



#read dataset
dataset_csgo = read_csv('C:/learnAI/Machine_Learning_Project-main/Machine_Learning_project/Project_1_CSGO_prediction/data/csgo.csv')

#data statistic
# report = ProfileReport(dataset_csgo, title="CSGO Prediction", explorative=True)
# report.to_file('csgo_prediction.html')

# create new data except 2 column team_a_rounds and team_b_rounds
new_dataset_csgo = dataset_csgo.drop(['team_a_rounds','team_b_rounds', 'day', 'month', 'year','date', 'wait_time_s','match_time_s'], axis=1)

#phan loai du lieu theo cot
numeric_feature = ['ping','kills','assists','deaths','mvps','hs_percent','points']
categories_feature = ['map']

#tai sao lai dung ColumnTransformer -> co the xu ly song song du lieu , tranh data leakage, duy tri cau truc du lieu (ban chi dinh ro cot nao dung bo xu ly du lieu nao giup code de doc )
preprocessing = ColumnTransformer(
    transformers=[
        #(ten tu dat, bo xu ly, danh sach cot can xu ly
        ('numeric_preprocessing', StandardScaler(), numeric_feature),
        ('categorical_preprocessing', OneHotEncoder(), categories_feature)
    ],
)


# train_test_split
target = 'result'
x = new_dataset_csgo.drop(target,axis=1)
y = new_dataset_csgo[target]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 42)


# data preprocessing
X_train = preprocessing.fit_transform(X_train)
X_test = preprocessing.transform(X_test)


# #training all model -> filter top 5 model have most performance
clf = LazyClassifier(verbose= 0, ignore_warnings=True, custom_metric = None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
# print(models)

# grid_search
#model logistic regression
param_grid_model_1 = {
    'penalty':['l1','l2', 'elasticnet'],
    'C' : [0.01, 0.1, 1, 10, 100],
}
grid_search_model_1 = GridSearchCV(estimator=LogisticRegression(random_state = 42), param_grid = param_grid_model_1,cv = 5, verbose = 2)
grid_search_model_1.fit(X_train, y_train)
# print(classification_report(y_test, grid_search_model_1.predict(X_test)))

"""
              precision    recall  f1-score   support

        Lost       0.72      0.77      0.74       225
         Tie       0.00      0.00      0.00        37
         Win       0.72      0.78      0.75       192

    accuracy                           0.71       454
   macro avg       0.48      0.52      0.50       454
weighted avg       0.66      0.71      0.68       454   

"""

#model Linear SVC
param_grid_model_2 = {
    'penalty':['l1','l2'],
    'loss': ['hinge', 'squared_hinge'],
    'C' : [0.01, 0.1, 1, 10, 100],
    'class_weight' : ['balanced']
}
grid_search_model_2 = GridSearchCV(estimator= LinearSVC(random_state=42), param_grid = param_grid_model_2, cv = 5, verbose = 2)
grid_search_model_2.fit(X_train, y_train)

#danh gia model Linear SVC
# print(classification_report(y_test, grid_search_model_2.predict(X_test)))

"""
              precision    recall  f1-score   support

        Lost       0.71      0.72      0.71       225
         Tie       0.19      0.27      0.22        37
         Win       0.78      0.70      0.74       192

    accuracy                           0.67       454
   macro avg       0.56      0.56      0.56       454
weighted avg       0.70      0.67      0.68       454
"""

# model RandomForestClassifier
param_grid_model_3 = {
    'criterion':['gini','entropy', 'log_loss'],
    'max_depth': [15, 20, 30],
    'class_weight' : ['balanced'],
}

grid_search_model_3 = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid = param_grid_model_3, cv = 5, verbose = 2)
grid_search_model_3.fit(X_train, y_train)

#danh gia mo hinh RandomForestClassifier
# print(classification_report(y_test, grid_search_model_3.predict(X_test)))
"""
              precision    recall  f1-score   support

        Lost       0.70      0.83      0.76       225
         Tie       0.33      0.05      0.09        37
         Win       0.78      0.73      0.76       192

    accuracy                           0.72       454
   macro avg       0.60      0.54      0.54       454
weighted avg       0.70      0.72      0.70       454
"""

# model Calibrated Classifier CV
param_grid_model_4 = {
    'method': ['sigmoid','isotonic'],
    'n_jobs': [-1]
}
grid_search_model_4 = GridSearchCV(estimator=CalibratedClassifierCV(), param_grid = param_grid_model_4, cv = 5, verbose = 2)
grid_search_model_4.fit(X_train, y_train)

# danh gia mo hinh Calibrated Classifier CV
# print(classification_report(y_test,grid_search_model_4.predict(X_test)))

"""
              precision    recall  f1-score   support

        Lost       0.72      0.79      0.75       225
         Tie       0.00      0.00      0.00        37
         Win       0.72      0.78      0.75       192

    accuracy                           0.72       454
   macro avg       0.48      0.52      0.50       454
weighted avg       0.66      0.72      0.69       454

"""

#model ExtraTrees Classifier
param_grid_model_5 = {
    'criterion':['gini','entropy', 'log_loss'],
    'max_depth': [15, 20, 30],
    'class_weight' : ['balanced'],
    'max_features' : ['sqrt', 'log2']
}
grid_search_model_5 = GridSearchCV(estimator=ExtraTreesClassifier(random_state=42), param_grid = param_grid_model_5, cv = 5, verbose = 2)
grid_search_model_5.fit(X_train, y_train)

#danh gia model ExtraTreesClassifier
# print(classification_report(y_test,grid_search_model_5.predict(X_test)))

"""
              precision    recall  f1-score   support

        Lost       0.69      0.78      0.73       225
         Tie       0.40      0.11      0.17        37
         Win       0.74      0.73      0.74       192

    accuracy                           0.71       454
   macro avg       0.61      0.54      0.55       454
weighted avg       0.69      0.71      0.69       454
"""

# -------> model has most performance is RandomForestClassifier
