import numpy as np
import pandas as pd
from dataset_loader import DatasetLoader
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
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
    x, Y = dataset.load_csv('./fgd_status_predition/dataset/train.csv', num_classes)

    x /= 255
    x = x.reshape(-1, 1, input_dim)
    print(x.shape)
    # x = add_lag_feature(x)
    Y = to_categorical(Y, num_classes)

    model = Sequential()
    model.add(LSTM(24, return_sequences=True, input_shape=(1, input_dim)))
    model.add(LSTM(16))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    optimizer = Adam(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, Y, validation_split = 0.2, epochs = 500, batch_size = 64)

    model.save('./fgd_prediction/fgd_lstm.h5')
