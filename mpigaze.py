from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io
import math
import os


def _extract_file_data(file_name):
    return scipy.io.loadmat(file_name, mat_dtype=True, squeeze_me=True)


class MPIGaze:
    data_dir = "C:/Users/Nirds2/Documents/Bhanuka/Datasets/MPIIGaze/Data/Normalized/"

    file_names = None

    df = None

    def __init__(self, data_dir=None):
        if not data_dir is None:
            self.data_dir = data_dir

    def _file_names(self):

        self.file_names = []
        directories = os.listdir(self.data_dir)
        for directory in directories:
            files = os.listdir(self.data_dir + directory)
            for file in files:
                self.file_names.append(self.data_dir + directory + "/" + file)  #
        # if self.file_names is None:
        #     self.file_names = ["day01.mat"]
        return self.file_names

    def _format(self):
        bins = np.array([-10.0, 10.0])
        self.df[:, 1] = self.df[:, 1] * 180 / math.pi
        self.df[:, 2] = self.df[:, 2] * 180 / math.pi
        self.df = np.hstack((self.df, np.digitize(self.df[:, 1], bins).reshape(-1, 1)))
        self.df = np.hstack((self.df, np.digitize(self.df[:, 2], bins).reshape(-1, 1)))
        self.df = np.hstack((self.df, (self.df[:, 3] * 3 + self.df[:, 4]).reshape(-1, 1)))
        # self.df = np.append(self.df, np.digitize(self.df[:, 1], bins), axis=0)
        # self.df = np.append(self.df, np.digitize(self.df[:, 2], bins), axis=0)


    def load_data(self):

        if self.df is None:
            self.read_data()

        return self.df

    def load_np(self, file_name='data.npy'):
        if self.df is None:
            self.df = np.load(file_name, allow_pickle=True)
            self._format()
        return self.df

    def load_train_test(self, dir=None, test_size=0.25):
        # if dir == 'H':
        #     return train_test_split(self.df[:, 0], self.df[:, 1], test_size=test_size)
        # return train_test_split(self.df[:, 0], self.df[:, 2], test_size=test_size)
        # if dir == 'H':
        #     return train_test_split(np.array(list(self.df[:, 0])), self.df[:, 1], test_size=test_size)
        # if dir == 'V':
        #     return train_test_split(np.array(list(self.df[:, 0])), self.df[:, 2], test_size=test_size)
        # else:
        #     return train_test_split(np.array(list(self.df[:, 0])), self.df[:, 1] * 3 + self.df[:, 2], test_size=test_size)
        return train_test_split(np.array(list(self.df[:, 0])), self.df[:, 1:].astype('float32'), test_size=test_size, random_state=42)

    def read_data(self):

        for file_name in self._file_names():
            print(file_name)
            data = _extract_file_data(file_name)

            right_gaze = data['data']['right'].item()['gaze'].item()
            right_images = data['data']['right'].item()['image'].item()

            if right_gaze.ndim == 1:
                right_gaze = right_gaze.reshape((1, 3))
                right_images = right_images.reshape((1, 36, 60))

            for i in range(0, len(right_gaze)):

                x = right_gaze[i][0]
                y = right_gaze[i][1]
                z = right_gaze[i][2]
                right_image = right_images[i]

                theta = math.asin(-y)
                phi = math.atan2(-x, -z)

                if self.df is None:
                    self.df = np.array([right_image, theta, phi]).reshape((1, 3))
                else:
                    self.df = np.append(self.df, [[right_image, theta, phi]], axis=0)

            left_gaze = data['data']['left'].item()['gaze'].item()
            left_images = data['data']['left'].item()['image'].item()
            if left_gaze.ndim == 1:
                left_gaze = left_gaze.reshape((1, 3))
                left_images = left_images.reshape((1, 36, 60))

            for i in range(0, len(left_gaze)):

                x = left_gaze[i][0]
                y = left_gaze[i][1]
                z = left_gaze[i][2]
                left_image = left_images[i]

                theta = math.asin(-y)
                phi = math.atan2(-x, -z)

                if self.df is None:
                    self.df = np.array([left_image, theta, phi]).reshape((1, 3))
                else:
                    self.df = np.append(self.df, [[left_image, theta, phi]], axis=0)
