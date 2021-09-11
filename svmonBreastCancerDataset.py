import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']
#classifier = svm.SVC(kernel="linear", C=2)
#classifier = svm.SVC(kernel="poly", degree=2) | 0.94

#Trying KNeighbourClassifier
#classifier = KNeighborsClassifier(n_neighbors=9) | 0.9736842105263158
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)