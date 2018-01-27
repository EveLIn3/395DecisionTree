"""
@author: rin
"""
import scipy.io as sio
from sklearn import tree
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=10)

data = sio.loadmat('cleandata_students.mat')
X = data['x']
Y = data['y']

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    dtree = tree.DecisionTreeClassifier().fit(X_train, Y_train)
    print(dtree.score(X_test, Y_test))