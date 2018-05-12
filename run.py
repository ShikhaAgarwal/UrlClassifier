from load_data import create_dataset
from classifier import initialize_classifier
from sklearn.metrics import *
import train

root = "/Users/shikha/UMass/summer2018/url_classifier/datasets/"
label = "Arts"
X, Y = create_dataset(root, label)
train_len = int(0.8 * X.shape[0])
X_train = X[0:train_len]
Y_train = Y[0:train_len]
X_test = X[train_len:]
Y_test = Y[train_len:]

print "Initializing classifier...."
clf = initialize_classifier()

print "Training cv...."
confusion, scores = train.train_cv(clf, X_train, Y_train)
print "Total tweets classified:", X.shape[0]
print "F1 Score:", sum(scores)/len(scores)
print "Confusion matrix:"
print confusion

# Re-train with all data
print "Training on all the dataset...."
train.train(clf, X_train, Y_train)
y_pred = train.predict(clf, X_test, Y_test)
print "F1 score on test data:", f1_score(Y_test, y_pred)
print "Confusion Matrix: "
print confusion_matrix(Y_test, y_pred)
print 'precision', precision_score(Y_test, y_pred)
print 'recall', recall_score(Y_test, y_pred)
print 'accuracy', accuracy_score(Y_test, y_pred)
