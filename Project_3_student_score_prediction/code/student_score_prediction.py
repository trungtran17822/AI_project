from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge

#read dataset
dataset_student_score = read_csv('C:/learnAI/Machine_Learning_Project-main/Machine_Learning_project/Project_3_student_score_prediction/data/Student_score.csv')

#create report data
# report = ProfileReport(dataset_student_score, title="Student Score Report", explorative=True)
# report.to_file('student_score_report.html')

#split data
target = 'writing score'
x = dataset_student_score.drop(target, axis=1)
y = dataset_student_score[target]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#category data : nominal feature, ordinal feature, numerical feature
numerical_feature = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
nominal_feature = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False)),
])

education_level = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
gender_values = ['male', 'female']
lunch_values = dataset_student_score['lunch'].unique()
test_values = dataset_student_score['test preparation course'].unique()
ordinal_feature = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(categories=[education_level, gender_values, lunch_values, test_values])),
])


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_feature, ['math score','reading score']),
    ('nom', nominal_feature, ['race/ethnicity']),
    ('ord', ordinal_feature, ['parental level of education', 'gender','lunch','test preparation course'])
])

#test data with LinearRegression
# reg = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', LinearRegression())
# ])
#
# reg.fit(X_train, y_train)

# y_predict = reg.predict(X_test)
# for i, j in zip(y_predict, y_test):
#     print('Predicted: {}, Actual: {}'.format(i, j))

#test data with all models (lazypredict)
# reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
# models,predictions = reg.fit(X_train, X_test, y_train, y_test)

#top 5 models have most performance : HuberRegressor, LassoCV, ElasticNetCV, BayesianRidge, RidgeCV
#model 1: HuberRegressor
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression_model_1', HuberRegressor()),
])

param_grid_model_1 = {
    'regression_model_1__epsilon': [1.35, 2.35],
    'regression_model_1__max_iter' : [100],
    'regression_model_1__alpha' : [0.001, 0.01],
    'regression_model_1__fit_intercept' : [True, False],
    'regression_model_1__warm_start': [True, False]
}

grid_search_model_1 = GridSearchCV(reg, param_grid=param_grid_model_1, cv=5, n_jobs=-1, verbose=2)
grid_search_model_1.fit(X_train, y_train)

#evaluate model HuberRegression
print('MAE: {}'.format(mean_absolute_error(y_test, grid_search_model_1.predict(X_test))))
print('RMSE: {}'.format(root_mean_squared_error(y_test, grid_search_model_1.predict(X_test))))
print('R2: {}'.format(r2_score(y_test, grid_search_model_1.predict(X_test))))

"""
MAE: 3.203427423529256
RMSE: 3.868200047964864
R2: 0.937917353700604
"""

#model 2: LassoCV
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression_model_2', LassoCV()),
])

param_grid_model_2 = {
    'regression_model_2__n_alphas':[100, 200, 300],
    'regression_model_2__fit_intercept':[True, False],
    'regression_model_2__precompute' :['auto',True,False],
    'regression_model_2__max_iter':[1000,2000,3000],
    'regression_model_2__copy_X': [True, False],
    'regression_model_2__selection':['cyclic','random'],
    'regression_model_2__random_state':[42],
}

grid_search_model_2 = GridSearchCV(reg, param_grid=param_grid_model_2, cv=5, n_jobs=-1, verbose=2)
grid_search_model_2.fit(X_train, y_train)

#evaluate model LassoCV
print('MAE: {}'.format(mean_absolute_error(y_test, grid_search_model_2.predict(X_test))))
print('RMSE: {}'.format(root_mean_squared_error(y_test, grid_search_model_2.predict(X_test))))
print('R2: {}'.format(r2_score(y_test, grid_search_model_2.predict(X_test))))

"""
MAE: 3.195715084611994
RMSE: 3.862136257590359
R2: 0.9381118426526013
"""

#model 3: ElasticNetCV
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression_model_3', ElasticNetCV()),
])

param_grid_model_3 = {
    'regression_model_3__l1_ratio':[0.1, 0.5, 0.7, 0.9],
    'regression_model_3__n_alphas':[100, 200, 300],
    'regression_model_3__fit_intercept':[True, False],
    'regression_model_3__precompute':['auto',True,False],
    'regression_model_3__max_iter':[1000,2000,3000],
    'regression_model_3__copy_X': [True, False],
    'regression_model_3__positive':[True, False],
    'regression_model_3__random_state':[42]
}
grid_search_model_3 = GridSearchCV(reg, param_grid=param_grid_model_3, cv=5, n_jobs=-1, verbose=2)
grid_search_model_3.fit(X_train, y_train)

#evaluate model ElasticNetCV
# print('MAE: {}'.format(mean_absolute_error(y_test, grid_search_model_3.predict(X_test))))
# print('RMSE: {}'.format(root_mean_squared_error(y_test, grid_search_model_3.predict(X_test))))
# print('R2: {}'.format(r2_score(y_test, grid_search_model_3.predict(X_test))))

"""
MAE: 3.1955751938225627
RMSE: 3.8630636639294904
R2: 0.9380821169481015
"""

#model 4: BayesianRidge
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression_model_4', BayesianRidge()),
])

param_grid_model_4 = {
    'regression_model_4__max_iter':[300],
    'regression_model_4__compute_score':[True, False],
    'regression_model_4__fit_intercept':[True, False],
    'regression_model_4__copy_X': [True, False],
    'regression_model_4__lambda_init':[1],
    'regression_model_4__alpha_init':[1]
}
grid_search_model_4 = GridSearchCV(reg, param_grid=param_grid_model_4, cv=5, n_jobs=-1, verbose=2)
grid_search_model_4.fit(X_train, y_train)

#evaluate model BayesianRidge
# print('MAE: {}'.format(mean_absolute_error(y_test, grid_search_model_4.predict(X_test))))
# print('RMSE: {}'.format(root_mean_squared_error(y_test, grid_search_model_4.predict(X_test))))
# print('R2: {}'.format(r2_score(y_test, grid_search_model_4.predict(X_test))))

"""
MAE: 3.2037467555675834
RMSE: 3.8700368202775315
R2: 0.9378583811770775
"""

# model 5: RidgeCV
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression_model_5', Ridge()),
])

param_grid_model_5 = {
    'regression_model_5__alphas':[0.1,1.0,10.0],
    'regression_model_5__fit_intercept':[True, False],
    'regression_model_5__scoring':[None],
    'regression_model_5__gcv_mode':['auto', 'svd','eigen'],
    'regression_model_5__alpha_per_target':[True, False],
    'regression_model_5__store_cv_results':[True, False]
}

grid_search_model_5 = GridSearchCV(reg, param_grid=param_grid_model_5, cv=3, n_jobs=-1, verbose=2)
grid_search_model_5.fit(X_train, y_train)

#evaluate model RidgeCV
print('MAE: {}'.format(mean_absolute_error(y_test, grid_search_model_5.predict(X_test))))
print('RMSE: {}'.format(root_mean_squared_error(y_test, grid_search_model_5.predict(X_test))))
print('R2: {}'.format(r2_score(y_test, grid_search_model_5.predict(X_test))))
"""
MAE: 3.202180153348796
RMSE: 3.8645604905905926
R2: 0.9380341248352577
"""

# => model LassoCV has most performance