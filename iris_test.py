import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Import the Perceptron class
from perceptron import Perceptron
from adaline import Adaline
from adalineSGD import AdalineSGD

# Load iris data from CSV file
file_path = r"C:\Python\ML\Perceptrons\iris\iris.data"

# There is no header in the CSV file, therefore 'header=None'
iris = pd.read_csv(file_path, header=None)

# Split to features (X) and labels (y)
X = iris.iloc[:, :-1].values  # All colums except the last one are featrures
y = iris.iloc[:, -1].values   # The last column is a label

# Change label to numerical values
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([label_map[label] for label in y])

# Let's make it binary classififcation by ignoring one label type
binary_filter = (y == 0) | (y == 1)
X = X[binary_filter]
y = y[binary_filter]

# Shuffle the data
#np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Initialization and perceptron training on train set
ppn = Perceptron(learning_rate=0.01, n_iters=10)
ppn.fit(X_train, y_train)

# Initialization and adaline training on train set
ppn_ada = Adaline(learning_rate=0.01, n_iters=10)
ppn_ada.fit(X_train, y_train)

# Initialization and adaline with SGD training on train set
ppn_ada_sgd = AdalineSGD(learning_rate=0.01, n_iters=10)
ppn_ada_sgd.fit(X_train, y_train)


# Predictions on test sets
y_pred = ppn.predict(X_test)
y_pred_ada = ppn_ada.predict(X_test)
y_pred_ada_sgd = ppn_ada_sgd.predict(X_test)

# accuracy of classification
accuracy = accuracy_score(y_test, y_pred)

accuracy_ada = accuracy_score(y_test, y_pred_ada)

accuracy_ada_sgd = accuracy_score(y_test, y_pred_ada_sgd)

# Display the consusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_ada = confusion_matrix(y_test, y_pred_ada)
cm_ada_sgd = confusion_matrix(y_test, y_pred_ada_sgd)

# Sformatowane wyniki
print("\n=== Model Evaluation Results ===\n")

# Perceptron
print(f"Perceptron Model:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

print("\n")

# Adaline
print(f"Adaline Model:")
print(f"Accuracy: {accuracy_ada:.4f}")
print("Confusion Matrix:")
print(cm_ada)

print("\n")

# Adaline with SGD
print(f"Adaline with SGD Model:")
print(f"Accuracy: {accuracy_ada_sgd:.4f}")
print("Confusion Matrix:")
print(cm_ada_sgd)
