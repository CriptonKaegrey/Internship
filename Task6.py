# Importing libraries in Python
import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Loading the iris dataset
iris = datasets.load_iris()

# Forming the iris dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# Label
y = iris.target
print('Labels are: {}, {} and {}'.format(iris.target_names[0], iris.target_names[1], iris.target_names[2]))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=1)

# Creating Decision tree classifier object
dtree = DecisionTreeClassifier(random_state=0)
print('Decision Tree Classifier Created')

# Training decision tree classifier
dtree.fit(X_train, y_train)
print('Data trained')

# Predicting the response for Test data
y_pred = dtree.predict(X_test)

# Checking Model Accuracy
print('Accuracy: {:.0%}'.format(accuracy_score(y_test, y_pred)))

# Visualizing the graph
dtree.fit(df, y)
dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data, feature_names=iris.feature_names,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())
print('Visualization complete')
