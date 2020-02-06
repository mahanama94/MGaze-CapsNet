import pandas as pd

# MPIIGaze
df = pd.read_csv('annotation.txt', sep=' ', header=None)

# Gaze position relative to camera
df[[26, 27, 28]].head()
