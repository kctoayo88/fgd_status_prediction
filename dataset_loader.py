import numpy as np
import pandas as pd

class DatasetLoader(object):
    def __init__(self):
        self.x = None
        self.Y = None

    def convert_label_from_zero(self):
        unique_Y = np.unique(self.Y)
        for y in self.Y:
            if y[0] == unique_Y[0]:
                y[0] = 0
            if y[0] == unique_Y[1]:
                y[0] = 1
            if y[0] == unique_Y[2]:
                y[0] = 2

    def load_csv(self, data_path, num_classes):
        raw_dataset = pd.read_csv(data_path)
        data = raw_dataset.iloc[1:, 2:]
        data = data[data['Y'] < num_classes + 1]

        self.x = (data.iloc[:, :-1]).to_numpy()
        print('The shape of feature data:', self.x.shape)

        self.Y = (data.iloc[:, -1:]).to_numpy()
        self.convert_label_from_zero()
        print('The shape of label data:', self.Y.shape)

        return self.x, self.Y

    def load_test_data(self, test_data_path):
        raw_dataset = pd.read_csv(test_data_path)
        testing_data = raw_dataset.iloc[1:, 2:]

        testing_data = testing_data.to_numpy()
        print('The shape of feature data:', testing_data.shape)
        
        return testing_data