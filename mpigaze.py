import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io
import math
from sklearn.model_selection import train_test_split

def _extract_file_data(file_name):
    return scipy.io.loadmat(file_name, mat_dtype=True, squeeze_me=True)


class MPIGaze:
    file_names = None

    df = None

    def _file_names(self):
        if self.file_names is None:
            self.file_names = ["day01.mat"]
        return self.file_names

    def load_data(self):

        if self.df is None:
            self.read_data()

        return self.df

    def read_data(self):

        for file_name in self._file_names():
            data = _extract_file_data(file_name)

            right_gaze = data['data']['right'].item()['gaze'].item()

            for i in range(0, len(right_gaze)):

                x = right_gaze[i][0]
                y = right_gaze[i][1]
                z = right_gaze[i][2]
                image = right_gaze[i]

                theta = math.asin(-y)
                phi = math.atan2(-x, -z)

                if self.df is None:
                    self.df = np.array([image, theta, phi]).reshape((1, 3))
                else:
                    self.df = np.append(self.df, [[image, theta, phi]], axis=0)

            left_gaze = data['data']['left'].item()['gaze'].item()

            for i in range(0, len(left_gaze)):

                x = left_gaze[i][0]
                y = left_gaze[i][1]
                z = left_gaze[i][2]
                image = left_gaze[i]

                theta = math.asin(-y)
                phi = math.atan2(-x, -z)

                if self.df is None:
                    self.df = np.array([image, theta, phi]).reshape((1, 3))
                else:
                    self.df = np.append(self.df, [[image, theta, phi]], axis=0)
