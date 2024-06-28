import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpi
import matplotlib.pyplot as plt

# https: // stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or -matplotlib-pyplot
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# mnist = fetch_openml("mnist_784", version=1)
X= mnist['data']
y = mnist['target']
pix = X[0]
# print(type(X))

finger_image = pix.reshape(28,28)
plt.imshow(finger_image, cmap="binary")
plt.axis("off")
plt.show()
