import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Dropout
from keras.applications import EfficientNetB0
from keras.utils import image_dataset_from_directory
import livelossplot as llp 
import matplotlib.pyplot as plt


data_dir=".../Datasets/flower_photos"

train_ds= image_dataset_from_directory(data_dir ,
 validation_split=0.2 ,
 subset="training" ,
 seed=123 ,
 label_mode="categorical" ,
 image_size=(150,150) ,
 batch_size=32)


val_ds= image_dataset_from_directory(data_dir , 
validation_split=0.2 ,
 subset="validation" ,
 seed=123 ,
  label_mode="categorical" , 
  image_size=(150,150) , 
  batch_size=32) 


pretrained_model = EfficientNetB0(include_top=False , weights="imagenet" , input_shape=(150,150,3) , 
pooling="avg" , classes=5 )

for layer in pretrained_model.layers :
    layer.trainable=False 


model=Sequential()
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128 , activation="relu"))
model.add(Dense(5 , activation="softmax"))

model.build(input_shape=(150,150,3))
model.summary()

plot_loss = llp.PlotLossesKeras()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history=model.fit(train_ds , batch_size=128, epochs=15, validation_data=val_ds)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()

model.save("flower_92%_accuracy.h5")
