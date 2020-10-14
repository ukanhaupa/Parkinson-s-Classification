import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
#%matplotlib inline
#from IPython.core.display import Image
#Image(filename='brain-activity.jpg')
parkinson_data = pd.read_csv("parkinsons_data.txt")
parkinson_data.head()
labels = parkinson_data.iloc[:, 0].values
features = parkinson_data.iloc[:, 1:].values
print(features.shape)
print(labels[0])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scled_features = scaler.fit_transform(features)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scled_features, labels, test_size=0.3)
print("X_train shape -- > {}".format(X_train.shape))
print("y_train shape -- > {}".format(y_train.shape))
print("X_test shape -- > {}".format(X_test.shape))
print("y_test shape -- > {}".format(y_test.shape))
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=300)
etc.fit(X_train, y_train)
print(etc.feature_importances_)
indices = np.argsort(etc.feature_importances_)[::-1]
plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w')
plt.title("Feature importances")
plt.bar(range(features.shape[1]), etc.feature_importances_[indices],
       color="r", align="center")
plt.xticks(range(features.shape[1]), indices)
plt.show()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN with k=5 got {}% accuracy on the test set.".format(accuracy_score(y_test, knn.predict(X_test))*100))
params_dict = {'n_neighbors':[3, 5, 9, 15], 'p':[1, 2, 3], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
gs = GridSearchCV(knn, param_grid=params_dict, verbose=10, cv=10)
gs.fit(X_train, y_train)
print(gs.best_estimator_)
new_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=1,
           weights='uniform')
new_knn.fit(X_train, y_train)
print("KNN - fine tuned, got {}% accuracy on the test set.".format(accuracy_score(y_test, new_knn.predict(X_test))*100))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic regression - default, got {}% accuracy on the test set.".format(accuracy_score(y_test, lr.predict(X_test))*100))
lr_tuned = LogisticRegression(C=1000, penalty='l2')
lr_tuned.fit(X_train, y_train)
print("Logistic regression - tuned, got {}% accuracy on the test set.".format(accuracy_score(y_test, lr_tuned.predict(X_test))*100))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print("Decision tree classifier, got {}% accuracy on the test set.".format(accuracy_score(y_test, dtc.predict(X_test))*100))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=350)
rfc.fit(X_train, y_train)
print("Random forest classifier, got {}% accuracy on the test set.".format(accuracy_score(y_test, rfc.predict(X_test))*100))
accuracy_tree = cross_val_score(dtc, scled_features, labels, scoring='accuracy', cv=10)
print(np.mean(accuracy_tree)*100)
