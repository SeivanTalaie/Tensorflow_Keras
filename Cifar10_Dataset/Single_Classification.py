#%% import libraries :
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import livelossplot as llp 
import seaborn as sns 
from tensorflow import keras 
from keras.models import Sequential , Model 
from keras.utils import to_categorical , plot_model 
from keras.layers import Dense , Flatten , Dropout , BatchNormalization , Conv2D , MaxPool2D , Input
from keras.layers import RandomFlip , RandomContrast , RandomZoom ,RandomRotation
from keras.datasets import cifar10 
from sklearn.metrics import confusion_matrix 
import pandas as pd

# %% Load the data :
(x_train , y_train) , (x_test , y_test) = cifar10.load_data()
num_classes=10

#%% Pre-processing the data : 
x_train=x_train.astype("float32") / 255.0
x_test=x_test.astype("float32") / 255.0

#%% One-hot-encoding : 
y_train = to_categorical(y_train , num_classes=num_classes)
y_test = to_categorical(y_test , num_classes=num_classes)

#%% build a model : 
input = Input(shape=(32,32,3))
x= Conv2D(256 , (3,3) , padding="valid" , activation="relu")(input) 
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Conv2D(128 , (3,3) , padding="valid" , activation="relu")(x) 
x= MaxPool2D()(x)
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Conv2D(128 , (3,3) , padding="valid" , activation="relu")(x) 
x= MaxPool2D()(x)
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Conv2D(128 , (3,3) , padding="valid" , activation="relu")(x) 
x= MaxPool2D()(x)
x=Dropout(0.2)(x)
x=BatchNormalization()(x)
x= Flatten()(x)
x= Dense(128 , activation="relu")(x)
x= Dense(64 , activation="relu")(x)
x= Dense(32 , activation="relu")(x)
output = Dense(10 , activation="softmax")(x)

model=Model(inputs = input , outputs = output) 
model.summary() 
plot_model(model , "fcn_cifar10.png" , show_shapes=True)

# %% Compile and fit the model :
plot_loss = llp.PlotLossesKeras()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history=model.fit(x_train, y_train, batch_size=128,
 epochs=15, validation_data=(x_test,y_test), callbacks=[plot_loss])

# %% Results :
import pandas as pd 
pd.DataFrame(history.history).plot()

# %% Confusion_matrix :
y_pred=model.predict(x_test)
sns.heatmap(confusion_matrix(np.argmax(y_pred,axis=1), np.argmax(y_test , axis=1)), cmap="Blues" , fmt="d" , annot=True)

# %% Examine the predictions :
R = 5
C = 5
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
Y_true = np.argmax(y_test, axis=1)
Y_pred_classes = np.argmax(y_pred, axis=1) 
fig, axes = plt.subplots(R, C, figsize=(12,12))
axes = axes.ravel()

for i in np.arange(0, R*C):
    axes[i].imshow(x_test[i])
    axes[i].set_title("True: %s \nPredict: %s" % (labels[Y_true[i]], labels[Y_pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1.5)
plt.show()



