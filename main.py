# Importing the libraries
from KNN import KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Importing the dataset
wine_data = pd.read_csv('wine.data', sep=",", header=None)
X = wine_data.iloc[:, 1:].values
y = wine_data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# creating and training KNN model
clf = KNN(k=3)
clf.fit(X_train, y_train)

# making predictions
predictions = clf.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)
