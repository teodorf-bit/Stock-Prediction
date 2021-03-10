'''
Stockpred: This python package is devoted to efficient implementation of machine learning algorithms
for predicting stock prices and evalauting trading strategies.

See README.md at https://github.com/teodorf-bit/Stock-Prediction
Author: Teodor Fredriksson, 2021

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as random
import sys, getopt, time, csv, torch, os, multiprocessing
import talib as ta
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, naive_bayes, neighbors
from sklearn import metrics
from scipy.stats import pearsonr

def SP_dir():

    d = os.path.dirname(os.path.realpath(__file__))
    if not os.access(d.os.W_OK):
        d = os.getcwd()

    return d


def create_dataset(raw_dataset):

    # Read in the data.
    dataset = pd.read_csv(raw_dataset+".csv")

    # Create the featutes.
    dataset['MACD'], NaN,  NaN = ta.MACD(dataset["Close"])
    dataset['CCI'] = ta.CCI(dataset["High"],dataset["Low"],dataset["Close"])
    dataset['ATR'] = ta.ATR(dataset['High'],dataset["Low"],dataset["Close"])
    dataset['upper'], dataset['middle'], dataset['lower'] = ta.BBANDS(dataset["Close"])
    dataset['EMA'] = ta.EMA(dataset['Close'])
    dataset['MA5'] = ta.MA(dataset['Close'], timeperiod=5)
    dataset['MA10'] = ta.MA(dataset['Close'], timeperiod=10)
    dataset['ROC'] = ta.ROC(dataset['Close'])
    dataset['SMI'], NaN = ta.STOCH(dataset['High'],dataset['Low'],dataset['Close'])

    # Save the features to a file.
    dataset.to_csv(raw_dataset+"_features.csv")
    dataset = dataset.dropna()
    return dataset

def regression(dataset, model):

    # Extract features and labels
    y = dataset['Close']
    X = dataset.drop(['Close','Date'], axis=1)

    # Split the data into training and test data.
    X_train = X.iloc[0:round(0.8*len(X))-1]
    X_test = X.iloc[round(0.8*len(X)):]

    y_train = y.iloc[0:round(0.8*len(X))-1]
    y_test = y.iloc[round(0.8*len(X)):]

    y_train = y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Define the model
    if model == 'linear':
        print("Use a linear model")
        reg = linear_model.LinearRegression()
    elif model == 'svm':
        print("Use a Support Vector Machine")
        reg = svm.SVR()
    elif model == 'nearest neighbor':
        print("K-nearest neighbors")
        reg = neighbors.KNeighborsRegressor(n_neighbors=4)
    elif model == 'decision tree':
        print("Use a K-nearest neighbors")
        reg = tree.DecisionTreeRegressor()
    else:
         print("Error! No algorithm was choosen")

    # Predict and evaluate
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y = y_test

    # Evaluate the mode
    MAPE = np.mean(abs((y-y_pred)/y))
    print('The Mean Average Procentage Error is', MAPE)
    R, NaN = pearsonr(y,y_pred)
    print('Pearsons correlation: %.3f' % R)
    U = np.square(np.mean((y-y_pred)**2))/(np.square(np.mean(y**2)) + np.square(np.mean(y_pred**2)))
    print('Theil U is', U)
    
    return y_pred, y

def invest(y_pred, y):
    P = 0
    for i in range(len(y_pred)-1):
        if y_pred[i+1] > y[i]:
            I = 1
            P = P-y[i]
        else:
            I = 0
            P = P+y[i]
    print('We just made', P)
    return P


