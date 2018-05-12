import numpy as np

from sklearn.metrics import *
from sklearn.model_selection import KFold

def train(clf, X_train, Y_train):
	clf.fit(X_train, Y_train)

def predict(clf, X_test, Y_test):
	y_pred = clf.predict(X_test)
	return y_pred

def train_cv(clf, X, Y):
	k_fold = KFold(n_splits=5, shuffle=True, random_state=3)
	split = k_fold.split(X)
	scores = []
	confusion = np.array([[0, 0], [0, 0]])
	for train_indices, test_indices in split:
	    X_train = X[train_indices]
	    Y_train = Y[train_indices]

	    X_val = X[test_indices]
	    Y_val = Y[test_indices]

	    train(clf, X_train, Y_train)
	    predictions = predict(clf, X_val, Y_val)

	    confusion += confusion_matrix(Y_val, predictions)
	    score = f1_score(Y_val, predictions)
	    scores.append(score)
	return confusion, scores
