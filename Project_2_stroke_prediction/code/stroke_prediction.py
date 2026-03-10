from pandas import read_csv
from ydata_profiling import ProfileReport
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import BaggingClassifier
#read dataset
dataset_stroke = read_csv('C:/learnAI/Machine_Learning_Project-main/Machine_Learning_project/Project_2_stroke_prediction/data/stroke_classification.csv')

#data statistic
# report = ProfileReport(dataset_stroke, title = 'Stroke_Prediction_Repot', explorative = True)
# report.to_file('strokeprediction.html')
dataset_stroke.dropna(inplace=True)

#data preprocessing
#split data
dataset_stroke.drop('pat_id',inplace=True, axis=1)

num_feature = ['smokes', 'age', 'hypertension', 'heart_disease', 'work_related_stress', 'urban_residence', 'avg_glucose_level', 'bmi']
cat_feature = ['gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_feature),
        ('cat', OneHotEncoder(), cat_feature),
    ]
)


target = 'stroke'
x = dataset_stroke.drop(target, axis=1)
y = dataset_stroke[target]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# X_train = preprocessor.fit_transform(X_train)
# X_test = preprocessor.transform(X_test)


# train dataset with all model
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

#find hyperparameter of model
#model 1: RandomForestClassifier
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
param_grid_model_1 = {
    'classifier__criterion': ['gini', 'entropy', 'log_loss'],
    'classifier__max_depth': [15, 20, 30],
    'classifier__class_weight': ['balanced'],
}
grid_search_model_1 = GridSearchCV(estimator=pipeline_rf,param_grid=param_grid_model_1, scoring='f1_macro' ,cv = 5, verbose=2)
grid_search_model_1.fit(X_train,y_train)

#evaluate model RandomForestClassifier
print(classification_report(y_test,grid_search_model_1.predict(X_test)))

"""
              precision    recall  f1-score   support

           0       0.96      0.93      0.94      1401
           1       0.11      0.15      0.13        72

    accuracy                           0.90      1473
   macro avg       0.53      0.54      0.54      1473
weighted avg       0.91      0.90      0.90      1473
"""

#model 2: KNeighbor Classifier
pipeline_kn = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', KNeighborsClassifier())
])
param_grid_model_2 = {
    'classifier__n_neighbors' :[3, 5],
    'classifier__weights' :['uniform', 'distance'],
    'classifier__algorithm':['auto', 'ball_tree', 'kd_tree','brute'],

}
grid_search_model_2 = GridSearchCV(estimator=pipeline_kn,param_grid=param_grid_model_2, scoring='f1_macro',cv = 5, verbose=2)

grid_search_model_2.fit(X_train,y_train)

#evaluate model KNeighborClassifier
print(classification_report(y_test,grid_search_model_2.predict(X_test)))

"""
           precision    recall  f1-score   support

           0       0.95      0.88      0.91      1401
           1       0.07      0.19      0.11        72

    accuracy                           0.84      1473
   macro avg       0.51      0.54      0.51      1473
weighted avg       0.91      0.84      0.87      1473
"""

#model 3: ExtraTreesClassifier
pipeline_et = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', ExtraTreesClassifier(random_state=42))
])
param_grid_model_3 = {
    'classifier__criterion':['gini', 'entropy', 'log_loss'],
    'classifier__max_depth':[15, 20, 30],
    'classifier__class_weight':['balanced'],
    'classifier__max_features':['sqrt', 'log2']
}
grid_search_model_3 = GridSearchCV(estimator=pipeline_et,param_grid=param_grid_model_3, scoring='f1_macro',cv = 5, verbose=2)
grid_search_model_3.fit(X_train,y_train)

#evaluate model ExtraTreesClassifier
print(classification_report(y_test,grid_search_model_3.predict(X_test)))
"""
              precision    recall  f1-score   support

           0       0.95      0.92      0.94      1401
           1       0.08      0.14      0.10        72

    accuracy                           0.88      1473
   macro avg       0.52      0.53      0.52      1473
weighted avg       0.91      0.88      0.90      1473
"""

#model 4: model NearestCentroid
pipeline_nc = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', NearestCentroid())
])
praram_grid_model_4 = {
    'classifier__metric':['euclidean','manhattan'],
    'classifier__priors' :['uniform','empirical']
}

grid_search_model_4 = GridSearchCV(estimator=pipeline_nc,param_grid=praram_grid_model_4, scoring='f1_macro',cv = 5, verbose=2)
grid_search_model_4.fit(X_train,y_train)

#evaluate model NearestCentroid
print(classification_report(y_test,grid_search_model_4.predict(X_test)))
"""
              precision    recall  f1-score   support

           0       0.98      0.79      0.87      1401
           1       0.15      0.72      0.25        72

    accuracy                           0.78      1473
   macro avg       0.56      0.75      0.56      1473
weighted avg       0.94      0.78      0.84      1473
"""


#model 5: Bagging Classifier
pipeline_bc = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', BaggingClassifier(random_state=42))
])
param_grid_model_5 = {
    'classifier__n_estimators':[10, 20,50],
    'classifier__max_features':[0.5, 0.8, 1.0]
}
grid_search_model_5 = GridSearchCV(estimator=pipeline_bc,param_grid=param_grid_model_5, scoring='f1_macro',cv = 5, verbose=2)
grid_search_model_5.fit(X_train,y_train)

#evaluate model Bagging Classifier
print(classification_report(y_test,grid_search_model_5.predict(X_test)))

"""
              precision    recall  f1-score   support

           0       0.95      0.97      0.96      1401
           1       0.14      0.11      0.12        72

    accuracy                           0.92      1473
   macro avg       0.55      0.54      0.54      1473
weighted avg       0.92      0.92      0.92      1473



"""
#-----> model NearestCentroid has most performance