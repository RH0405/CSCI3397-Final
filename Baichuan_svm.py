import pandas as pd
import numpy as np
from sklearn.svm import SVC
import keras
from keras import Sequential, Model, layers

def read_xlsx():
    data = pd.read_excel('Adults_NV800_avg.xlsx')

    r = {}
    for d in data.values:
        if d[2] not in r.keys():
            r[d[2]] = [[], []]
        if d[1] == 'Noun':
            y = 0
        elif d[1] == 'Verb':
            y = 1
        else:
            raise RuntimeError('unknow type')
        x = np.array(d[3:].tolist())
        # x = np.reshape(x, [-1, 16]).T
        r[d[2]][0].append(x)
        r[d[2]][1].append(y)
    return r

def splite(r):

    patients = list(r.keys())

    def getData(patients):
        X, y = [], []
        for patient in patients:
            pdata = r[patient]
            X = X + pdata[0]
            y = y + pdata[1]

        X = np.array(X)
        y = np.array(y)
        X[np.isnan(X)] = 0
        return X, y

    trainX, trainY = getData(patients[:17])
    valX, valY = getData(patients[17:])

    return trainX, trainY, valX, valY


def build_model():

    model = Sequential()
    model.add(layers.LSTM(64, input_shape=(16, 30), return_sequences=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':

    data = read_xlsx()

    trainX, trainY, valX, valY = splite(data)

    # model = build_model()

    svc = SVC()
    svc.fit(trainX, trainY)
    score = svc.score(valX, valY)
    print('accuracy: ', score)

