# Getting a best separate for the dataset. Used for binary classification

import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import cross_validate, train_test_split

import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.columns = ['id','clump_thickness','unif_cell_size','unif_cell_shape','marg-adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses','class']
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)