import numpy as np
import pandas as pd
from dataset_loader import DatasetLoader
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam

def buildTrain(train, label, pastDay = 10):
    x_train, Y_train = [], []
    for i in range(train.shape[0] - pastDay):
        x_train.append(train[i : i + pastDay])
        Y_train.append(label[i : i + pastDay])
    return np.array(x_train), np.array(Y_train)

def add_lag_feature(x):
    x = x.tolist()
    for i in range(len(x)):
        if i == 0:
            lag_feature = Y[i][0]
        else:
            lag_feature = Y[i-1][0]
        x[i].append(lag_feature)
    x = np.array(x)
    print('The shape of feature data (including lag feature):', x.shape)

    return x

if __name__ == '__main__':
    num_classes = 3
    learning_rate = 1e-3

    dataset = DatasetLoader()
    x, Y = dataset.load_csv('./dataset/train.csv', num_classes)

    x /= 255
    # x = add_lag_feature(x)
    x, Y = buildTrain(x, Y)
    print('The shape of training data:', x.shape)
    print('The shape of training label:', Y.shape)
    Y = to_categorical(Y, num_classes)

    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    optimizer = Adam(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, Y, validation_split = 0.2, epochs = 500, batch_size = 1024)

    model.save('./fgd_lstm.h5')
