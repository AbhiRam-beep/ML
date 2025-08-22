import tensorflow as tf 
from tensorflow.keras import layers,models 

(xtrain,ytrain),(xtest,ytest)=tf.keras.datasets.mnist.load_data() #a popular dataset that contains handwritten digits
xtrain,xtest=xtrain/255.0,xtest/255.0; #normalizing
xtrain=xtrain.reshape(-1,28*28)
xtest=xtest.reshape(-1,28*28) #flattening images with 28*28 dimension (original mnist dims)

model = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='linear')
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy']) 
#compiling the model using softmax activation function 
model.fit(xtrain,ytrain,epochs=5,batch_size=64,validation_split=0.1) #5 training iterations,64 images at a time,10% data used for validation

loss,accuracy = model.evaluate(xtest,ytest)
print(f'Test Accuracy: {accuracy*100:.2f}%')

logits = model(xtest[:5]) #retrieving the logits or 'z' values for 5 test cases
print("Logits for first 5 test images:\n", logits)

probabilities = tf.nn.softmax(logits, axis=1)  # apply softmax along classes to get the probabilities
print("Probabilities for first image:\n", probabilities[0])
# convert logits to digit labels by getting maximum logits' value for each input image
labels = tf.argmax(logits, axis=1).numpy()  
print("Predicted digits for first 5 test images:", labels)
