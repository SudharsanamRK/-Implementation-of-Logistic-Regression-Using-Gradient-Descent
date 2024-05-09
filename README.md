# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## Aim
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
```

## Program

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output
## Array Value of x
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/ee04b5ba-b062-4370-84ab-6d795bb72f34)

## Array Value of y
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/03c79e5a-ac27-4f1c-8eed-90954c7eb53d)


## Exam 1 - score graph
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/ede9536f-2986-490b-b099-480a60ed65cf)

## Sigmoid function graph
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/6b747680-4bb6-430b-bcd4-869494fb4452)

## X_train_grad value
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/c85fe4d7-5a75-49e5-8aad-e2c58a496433)

## Y_train_grad value
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/03796c00-1e70-4bcb-8cdd-620f49bd96e0)

## Print res.x
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/73637029-6232-4b4f-8977-917820af263b)


## Decision boundary - graph for exam score
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/ac0481dc-c588-49bb-b47d-64dee3815ae2)

## Proability value
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/e3b4c950-00a7-4910-96c3-6fef8e82f9ca)

## Prediction value of mean
![image](https://github.com/SudharsanamRK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115523484/764dafd0-318c-486c-befa-a7b09485e7fc)


## Result
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

