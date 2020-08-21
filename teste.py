import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import data
from skimage.color import rgb2gray, rgba2rgb

import numpy as np

import pandas as pd

teste = mpimg.imread('teste3.png')

x = pd.read_csv('train.csv', delimiter=',', header=0, dtype=float)

x_2 = x.drop(['label'], axis=1)
y = x_2.values

k = []
for i in y[45]:
    k.append(255 - i)

grayscale = rgb2gray(teste)

res2 = []
i = 0
while i < 28:
    linha = []
    j = 0
    res = []
    while j < 28:
        linha.append(y[45][i*28 + j])
        j += 1
    res.append(linha)
    res2.append(res)
    i += 1

y = np.concatenate((res2[0], res2[1], res2[2], res2[3], res2[4], res2[5], res2[6], res2[7], res2[8], res2[9],
                      res2[10], res2[11], res2[12], res2[13], res2[14], res2[15], res2[16], res2[17], res2[18],
                      res2[19], res2[20], res2[21], res2[22], res2[23], res2[24], res2[25], res2[26], res2[27]))

res2 = []
i = 0
while i < 28:
    linha = []
    j = 0
    res = []
    while j < 28:
        linha.append(k[i*28 + j])
        j += 1
    res.append(linha)
    res2.append(res)
    i += 1

k = np.concatenate((res2[0], res2[1], res2[2], res2[3], res2[4], res2[5], res2[6], res2[7], res2[8], res2[9],
                      res2[10], res2[11], res2[12], res2[13], res2[14], res2[15], res2[16], res2[17], res2[18],
                      res2[19], res2[20], res2[21], res2[22], res2[23], res2[24], res2[25], res2[26], res2[27]))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(y, cmap=plt.cm.gray)
ax[0].set_title("Original")
ax[1].imshow(k, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()
