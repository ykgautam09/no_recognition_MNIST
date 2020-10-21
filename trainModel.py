import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("Dataset_MNIST/train.csv")
x_train = data.iloc[0:, 1:].values
y_train = data.iloc[0:, 0].values

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f, protocol=2)
