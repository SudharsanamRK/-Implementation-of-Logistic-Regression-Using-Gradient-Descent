# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## Aim
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary python packages
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value
```

## Program

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
```

```python
# Importing necessary libraries
import pandas as pd
import numpy as np

# Reading data
data = pd.read_csv("/content/Placement_Data (1).csv")
data1 = data.copy()

# Dropping unnecessary columns
data1 = data.drop(['sl_no', 'salary'], axis=1)

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Extracting features and target variable
X = data1.iloc[:, :-1]
Y = data1["status"]

# Initializing parameters
theta = np.random.randn(X.shape[1])

# Defining sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Defining loss function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Defining gradient descent function
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# Training the model
theta = gradient_descent(theta, X, Y, alpha=0.01, num_iterations=1000)

# Making predictions
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

# Calculating accuracy
accuracy = np.mean(y_pred.flatten() == Y)
print("Accuracy:", accuracy)

# Displaying predictions
print("Predicted:\n", y_pred)
print("Actual:\n", Y.values)

# Making predictions on new data
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_pred_new = predict(theta, xnew)
print("Predicted Result:", y_pred_new)

```
## Output
## Accuracy , actual and predicted values:

![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/eb850aee-2335-4f44-972b-3918adb147bf)

## Predicted result:
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/2945c8a6-1679-4b6b-b3bd-17f528c8f61a)


## Result
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
