
import numpy as np

# sigmoid
def sigmoid(z_wb):
    return 1 / (1 + np.exp(-z_wb))

# loss for one sample
def loss(f_wb, y):
    return -(y*np.log(f_wb) + (1-y)*np.log(1-f_wb))

# cost for all samples
def cost(X, Y, w, b):
    m = X.shape[1]
    z_wb = np.dot(w.T, X) + b
    f_wb = sigmoid(z_wb)
    return (1/m) * np.sum(loss(f_wb, Y))

# gradient descent
def gradient_descent(X, Y, w, b, alpha, iters):
    m = X.shape[1]
    for i in range(iters):
        z_wb = np.dot(w.T, X) + b
        f_wb = sigmoid(z_wb)
        dw = (1/m) * np.dot(X, (f_wb - Y).T)
        db = (1/m) * np.sum(f_wb - Y)
        w -= alpha * dw
        b -= alpha * db
    return w, b

# gradient descent with L2 regularization
def gradient_descent_reg(X, Y, w, b, alpha, iters, lambda_):
    m = X.shape[1]
    for i in range(iters):
        z_wb = np.dot(w.T, X) + b
        f_wb = sigmoid(z_wb)
        dw = (1/m) * np.dot(X, (f_wb - Y).T) + (lambda_/m)*w
        db = (1/m) * np.sum(f_wb - Y)
        w -= alpha * dw
        b -= alpha * db
    return w, b

# prediction
def predict(X, w, b):
    z_wb = np.dot(w.T, X) + b
    f_wb = sigmoid(z_wb)
    return (f_wb > 0.5).astype(int)

# demo data
X = np.array([[1,2,3,4],[2,3,4,5]])   # shape (2,4)
Y = np.array([[0,0,1,1]])             # shape (1,4)

# initialize
w = np.zeros((X.shape[0],1))
b = 0

# train without regularization
w1, b1 = gradient_descent(X, Y, w.copy(), b, alpha=0.1, iters=1000)
pred1 = predict(X, w1, b1)

# train with regularization
w2, b2 = gradient_descent_reg(X, Y, w.copy(), b, alpha=0.1, iters=1000, lambda_=0.7)
pred2 = predict(X, w2, b2)

print("Predictions without regularization:", pred1)
print("Predictions with regularization:", pred2)
print("Weights without reg:", w1.T)
print("Weights with reg:", w2.T)
