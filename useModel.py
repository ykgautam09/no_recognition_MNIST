import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sklearn  
# from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import pickle

# data =pd.read_csv("Dataset_MNIST/train.csv")
# x_train=data.iloc[0:,1:].values
# y_train=data.iloc[0:,0].values

# #x_test=data.iloc[30000:,1:].values
# #y_test=data.iloc[30000:,0].values

# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(x_train,y_train)
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

filepath = 'testImages/mnist_data4.png'
img = np.array(Image.open(filepath))

plt.imshow(255 - img, cmap='gray')
plt.show()
print(model.predict([img.reshape(784)])[0])
