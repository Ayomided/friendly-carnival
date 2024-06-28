from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split

iris_df = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_df.data, iris_df.target, test_size=33, random_state=0)

clf = neighbors.KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
