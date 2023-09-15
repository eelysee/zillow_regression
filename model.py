import pandas as pd
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate as ev


def scale_train_val(X_train,X_val):
    '''
    scales train, val, test
    '''
    mms = MinMaxScaler()
    X_train[['bath', 'bed', 'sqft']] = mms.fit_transform(X_train[['bath', 'bed', 'sqft']])
    X_val[['bath', 'bed', 'sqft']] = mms.transform(X_val[['bath', 'bed', 'sqft']])
    return X_train, X_val


def polynomial_regression_model(X_train, y_train, X_val, y_val):
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    yhat_train = model.predict(X_train_poly)
    yhat_val = model.predict(X_val_poly)
    
    train_evals = ev.calculate_metrics(y_train, yhat_train)
    val_evals = ev.calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return X_testmodel