import numpy as np
from sklearn.model_selection import train_test_split
import os

file_names = ['columbia/' + file  for file in os.listdir('columbia')]

class ColumbiaGaze:

    df = None

    def _format(self):
        bins = np.array([-10.0, 10.0])
        self.df = np.hstack((self.df, np.digitize(self.df[:, 1], np.array([0.0])).reshape(-1, 1)))
        self.df = np.hstack((self.df, np.digitize(self.df[:, 2], bins).reshape(-1, 1)))
        self.df = np.hstack((self.df, (self.df[:, 3] * 3 + self.df[:, 4]).reshape(-1, 1)))

    def load_np(self):
        if self.df == None:
            self.df = np.load('columbia-gaze.npy', allow_pickle=True)
            self._format()
        return self.df

    def load_train_test(self, test_size=0.25):

        return train_test_split(np.array(list(self.df[:, 0])), self.df[:, 1:].astype('float32'), test_size=test_size, random_state=42)