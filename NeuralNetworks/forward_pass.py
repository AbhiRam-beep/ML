import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import Sequential 

Xtrain = np.array([0,1,2,3,4,5],dtype=np.float32).reshape(-1,1) 
Ytrain = np.array([0,0,0,1,1,1],dtype=np.float32).reshape(-1,1)

model = Sequential([
    tf.keras.layers.Dense(1,input_dim=1,activation='sigmoid',name='L1')
])

_ = model(Xtrain) #default inital weight and bias

layer1 = model.get_layer('L1')
w,b=layer1.get_weights()
print("w = ",w," b = ",b)

set_w = np.array([[2.0]])
set_b = np.array([-4.5]) #manually setting weight and bias

layer1.set_weights([set_w,set_b])

a1 = model.predict(Xtrain,verbose=0)
print(" Prediction = ",a1)
