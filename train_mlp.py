import numpy as np
import pandas as pd
from dataset_loader import DatasetLoader
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
    input_dim = 24
    num_classes = 3
    learning_rate = 1e-4

    dataset = DatasetLoader()
    x, Y = dataset.load_csv('./fgd_prediction/dataset/train.csv', num_classes)

    x /= 255
    # x = add_lag_feature(x)
    Y = to_categorical(Y, num_classes)

    model = Sequential()
    model.add(Dense(24, input_dim = input_dim, activation='relu'))
    model.add(Dense(12, activation ='relu'))
    model.add(Dense(8, activation ='relu'))
    model.add(Dense(num_classes, activation ='softmax'))
    model.summary()

    optimizer = Adam(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, Y, validation_split = 0.2, epochs = 500, batch_size = 128)

    model.save('./fgd_prediction/mlp/fgd_mlp.h5')
