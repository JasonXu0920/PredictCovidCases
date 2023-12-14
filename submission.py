import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import math


## Project-Part1
def predict_COVID_part1(svm_model, train_df, train_labels_df, past_cases_interval, past_weather_interval, test_feature):
    x_train = pd.DataFrame()
    for i, row in train_df[30:].iterrows():
        index = i - 30
        for j in range(past_weather_interval, 0, -1):
            x_train.loc[index, ['max_temp' + '-' + str(j)]] = train_df.loc[i - j, 'max_temp']
        for j in range(past_weather_interval, 0, -1):
            x_train.loc[index, ['max_dew' + '-' + str(j)]] = train_df.loc[i - j, 'max_dew']
        for j in range(past_weather_interval, 0, -1):
            x_train.loc[index, ['max_humid' + '-' + str(j)]] = train_df.loc[i - j, 'max_humid']
        for j in range(past_cases_interval, 0, -1):
            x_train.loc[index, ['dailly_cases' + '-' + str(j)]] = train_df.loc[i - j, 'dailly_cases']

    cols = list(x_train.columns)

    y_train = train_labels_df.loc[30:, 'dailly_cases']

    x = x_train.values
    y = y_train.values

    svm_model.fit(x, y)

    test_feature = test_feature[1:]
    test = []
    for i in test_feature.index:
        if i in cols:
            test.append(test_feature[i])

    pre = svm_model.predict([test])

    return int(math.floor(pre[0]))



## Project-Part2
def predict_COVID_part2(train_df, train_labels_df, test_feature):
    past_cases_interval = 15
    past_weather_interval = 1

    svm_model = SVR()
    svm_model.set_params(**{'kernel': 'poly', 'degree': 1, 'C': 6000,
                            'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})

    x_train = pd.DataFrame()
    for i, row in train_df[30:].iterrows():
        index = i - 30
        for j in range(past_weather_interval, 0, -1):
            x_train.loc[index, ['min_temp' + '-' + str(j)]] = train_df.loc[i - j, 'min_temp']

        for j in range(past_cases_interval, 0, -1):
            x_train.loc[index, ['dailly_cases' + '-' + str(j)]] = train_df.loc[i - j, 'dailly_cases']

    cols = list(x_train.columns)

    y_train = train_labels_df.loc[30:, 'dailly_cases']

    x = x_train.values
    y = y_train.values

    svm_model.fit(x, y)

    test_feature = test_feature[1:]
    test = []
    for i in test_feature.index:
        if i in cols:
            test.append(test_feature[i])

    pre = svm_model.predict([test])

    return int(math.floor(pre[0]))


