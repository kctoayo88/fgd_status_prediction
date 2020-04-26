import numpy as np
import pandas as pd
from dataset_loader import DatasetLoader
from keras.models import load_model

dataset = DatasetLoader()
x = dataset.load_test_data('./fgd_prediction/dataset/test.csv')
x /= 255

model = load_model('./fgd_prediction/model.h5')

pred = model.predict(x)

index = 0
for result in pred:
    print('index:{}, class:{}'.format(index, np.argmax(result)))
    index += 1