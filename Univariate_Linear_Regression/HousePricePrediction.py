import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#extracting data from csv file
data = pd.read_csv("~/python/ML/UnivariateLinearRegression/house_prices_dataset.csv")
x = np.array(data['area'].values,dtype=np.float64)
y = np.array(data['price'].values,dtype=np.float64)

w = 0.0
b = 0.0

#Calculating cost function with initialized values w and b
def computeCost(x, y, w, b):
    m = len(x)
    return np.sum((w*x + b - y)**2) / (2*m)

#calculating gradient ∂J(w,b)/∂w and ∂J(w,b)/∂b
def gradient(w, b, x, y):
    m = len(x)
    err = w*x + b - y
    dj_dw = np.sum(err * x) / m
    dj_db = np.sum(err) / m
    return dj_dw, dj_db

#calculating gradient descent until w and b are saturated at global minimum

def gradientDescent(x, y, w, b, num_iters):
    alpha = 0.001 # learning rate
    jHist = []
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient(w, b, x, y)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db 
        if i % 1000 == 0:
             print(f"Iter {i}, w: {w}, b: {b}")
        J = computeCost(x, y, w, b)
        jHist.append(J)
    
    return w, b, jHist


xm = np.mean(x)
xs = np.std(x)
ym = np.mean(y)
ys = np.std(y)

# Normalize for gradient descent
x = (x - xm) / xs
y = (y - ym) / ys

# Train gradient descent
w_final, b_final, J = gradientDescent(x, y, w, b, 100000)
print(f"Final cost J: {J[-1]:.2e}")

area = float(input("Enter house area: "))
a_norm = (area - xm) / xs
p_norm = w_final * a_norm + b_final
p = p_norm * ys + ym
print(f"Predicted house price: {p:.2f}")

yp_norm = w_final * x + b_final
yp = yp_norm * ys + ym

plt.scatter(x * xs + xm, y * ys + ym, label='Actual data')  # denormalizing
plt.plot(x * xs + xm, yp, color='red', label='Regression line')
plt.scatter(area, p, color='lime', s=100, label='My prediction')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()


