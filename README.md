# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![image](https://github.com/vinushcv/mnist-classification/assets/113975318/d44d4e64-dc8c-47b0-a2de-3193e83d552d)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot

### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input

## PROGRAM
### Name:vinush.cv
### Register Number:212222230176
```python

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train.shape

x_test.shape


singleimage=x_train[200]


singleimage.shape

plt.imshow(singleimage)

y_train.shape

x_train.min()

x_train.max()

x_train_scaled=x_train/255
x_test_scaled=x_test/255

x_train_scaled.min()


x_test_scaled.max()

y_train[0]

y_train_ohe=utils.to_categorical(y_train,10)
y_test_ohe=utils.to_categorical(y_test,10)

y_train_ohe.shape

single_image = x_train[500]
plt.imshow(single_image,cmap='gray')

y_train_ohe[500]

X_train_scaled = x_train_scaled.reshape(-1,28,28,1)
X_test_scaled = x_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=16, kernel_size=(9,9), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(65,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_ohe, epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_ohe))

metrics = pd.DataFrame(model.history.history)

metrics.head()

print("Name:Vinush.CV Reg.No:212222230176 ")
metrics[['accuracy','val_accuracy']].plot()

print("Name:Vinush.CV Reg.No:212222230176 ")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("Name:Vinush.CV Reg.No:212222230176 ")
print(confusion_matrix(y_test,x_test_predictions))

print("Name:Vinush.CV Reg.No:212222230176 ")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('9.png')

img

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot:

![image](https://github.com/vinushcv/mnist-classification/assets/113975318/e0f651e6-7ff0-4a57-9598-eb9c82048eca)


![image](https://github.com/vinushcv/mnist-classification/assets/113975318/86d67214-36ae-4173-a1bd-6e39ef992643)


### Classification Report:

![image](https://github.com/vinushcv/mnist-classification/assets/113975318/b0a586fa-6859-48e0-afb7-5a781d4b81be)



### Confusion Matrix:

![image](https://github.com/vinushcv/mnist-classification/assets/113975318/40eea0a0-d98b-4d25-aab2-9594e7485456)


### New Sample Data Prediction:

![image](https://github.com/vinushcv/mnist-classification/assets/113975318/0342e4cd-276e-41b0-a8fd-e4f06833f970)


![image](https://github.com/vinushcv/mnist-classification/assets/113975318/54df308f-26a3-4f74-86b6-71040d52e01a)




## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.


