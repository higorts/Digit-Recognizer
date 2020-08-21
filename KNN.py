from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

import pandas as pd

data = pd.read_csv('train.csv', delimiter=',', header=0)
print(data)
x_data = data.drop(['label'], axis=1)
x_data_vector = x_data.values

y_data = data['label']
y_data_vector = y_data.values

x_train, x_test, y_train, y_test = train_test_split(x_data_vector, y_data_vector, test_size=0.01, random_state=42)

clf = NearestCentroid()
clf.fit(x_train, y_train)

predc = clf.predict(x_test)

print(predc, y_test, accuracy_score(y_test, predc))