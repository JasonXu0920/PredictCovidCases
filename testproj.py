import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from math import floor


## Read training data
train_file = './data/COVID_train_data.csv'
train_df = pd.read_csv(train_file)
#(192, 18)

## Read Training labels
train_label_file = './data/COVID_train_labels.csv'
train_labels_df = pd.read_csv(train_label_file)
#(192, 2)


## Read testing Features
test_fea_file = './data/test_features.csv'
test_features = pd.read_csv(test_fea_file)
#(20, 511)



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
        #for j in range(past_weather_interval, 0, -1):
        #    x_train.loc[index, ['avg_pressure' + '-' + str(j)]] = train_df.loc[i - j, 'avg_pressure']
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

    return int(np.floor(pre[0]))


## Generate Prediction Results
predicted_cases_part2 = []
for idx in range(len(test_features)):
    test_feature = test_features.loc[idx]
    prediction = predict_COVID_part2(train_df, train_labels_df, test_feature)
    predicted_cases_part2.append(prediction)

## MeanAbsoluteError Computation...!

test_label_file ='./data/COVID_test_labels.csv'
test_labels_df = pd.read_csv(test_label_file)
ground_truth = test_labels_df['dailly_cases'].to_list()

MeanAbsError = mean_absolute_error(predicted_cases_part2, ground_truth)
print('MeanAbsError = ', MeanAbsError)



