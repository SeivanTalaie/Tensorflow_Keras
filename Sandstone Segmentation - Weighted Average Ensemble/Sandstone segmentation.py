"""
                 (Sandstone segmentation)

"""

###################### Libraries ######################

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv 
import keras 
from sklearn.model_selection import train_test_split 
from keras.models import load_model 
from keras.utils import to_categorical
from tqdm import tqdm
import segmentation_models as sm
import pandas as pd 
import glob
from keras.metrics import MeanIoU
from sklearn.preprocessing import LabelEncoder 


################## Create a dataset ###################

Data_dir=".../path/to/save/directory/"
img_list=glob.glob(Data_dir + "image/*.jpg")
image_dataset=[]
for i in tqdm(sorted(img_list)):
    image=cv.imread(i)
    image_dataset.append(image)
    
mask_list=glob.glob(Data_dir + "mask/*.jpg")
mask_dataset=[]
for i in tqdm(sorted(mask_list)):
    mask=cv.imread(i,0)
    mask_dataset.append(mask)
    
image_dataset=np.array(image_dataset)
mask_dataset=np.array(mask_dataset)
mask_dataset=np.expand_dims(mask_dataset, axis=3)


##################### Sanity check ####################

rand_num=np.random.randint(0,image_dataset.shape[0])
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(image_dataset[rand_num])
plt.subplot(122)
plt.imshow(mask_dataset[rand_num])
plt.show()


##################### Label Encoder ###################

print(np.unique(mask_dataset[rand_num]))

labelencoder = LabelEncoder()
training_mask=labelencoder.fit_transform(mask_dataset.reshape(-1,1))
training_mask=training_mask.reshape(mask_dataset.shape)

print(np.unique(training_mask[rand_num]))


################### Split the data ####################

x_train, x_test, y_train, y_test = train_test_split(image_dataset , training_mask , 
                                                    test_size=0.1 , random_state=49)

print(f"x_train shape : {x_train.shape}",
      f"x_test.shape {x_test.shape}:",
      f"y_train shape : {y_train.shape}", 
      f"y_test shape : {y_test.shape}", sep=("\n"))


#################### Sanity check #####################

rand_num1 = np.random.randint(0 , x_train.shape[0])

plt.figure(figsize=(10,10), dpi=170,)
plt.subplot(1,2,1)
plt.imshow(x_train[rand_num1])
plt.title("image")

plt.subplot(1,2,2)
plt.imshow(y_train[rand_num1])
plt.title("mask")
plt.show()


################# Data preprocessing ##################

backbone="efficientnetb0"
input_preprocessing = sm.get_preprocessing(backbone)

preprocessed_x_train=input_preprocessing(x_train)
preprocessed_x_test=input_preprocessing(x_test)

y_train=to_categorical(y_train , num_classes=4)
y_test=to_categorical(y_test , num_classes=4)


################ Define loss and metric ###############

loss="categorical_crossentropy" 

Dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = Dice_loss + (1 * focal_loss)

focal_dice_loss=sm.losses.categorical_focal_dice_loss

metric=[sm.metrics.IOUScore(threshold=0.5)]


################## Define a callback ##################

filepath="D:/Faradars/Model weights/Sandstone_weights/sandstone-effb0--{epoch}--{val_iou_score}.hdf5"
checkpoint=keras.callbacks.ModelCheckpoint(filepath, monitor="val_iou_score",
                                            save_best_only=True, verbose=1, mode="max")


############# Build and compile the model #############

model= sm.Unet(backbone,
               classes=4,
               encoder_weights="imagenet",
               input_shape=(128,128,3),
               activation="softmax")
model.summary()

lr=0.001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=[focal_dice_loss],
              metrics=[metric])


#################### Fit the model ####################

history = model.fit(preprocessed_x_train, y_train, epochs=100,
                    validation_data=(preprocessed_x_test,y_test),
                    batch_size=32, callbacks=[checkpoint])
                    
# model.save("sandstone_effb0_last_epoch.hdf5")
pd.DataFrame(history.history).plot()
plt.show()


############ Model performance evaluation ############# 

my_model = load_model("D:\Faradars\Model weights\Sandstone_weights/sandstone-effb0--53--0.9253259897232056.hdf5",compile=False)

y_pred=my_model.predict(preprocessed_x_test)
y_pred_argmax=np.argmax(y_pred , axis=3)
y_test_argmax=np.argmax(y_test , axis=3)

random_img=np.random.randint(0,x_test.shape[0])
plt.figure(figsize=(10,10), dpi=170)
plt.subplot(131)
plt.imshow(x_test[random_img])
plt.title("image")

plt.subplot(132)
plt.imshow(y_test_argmax[random_img])
plt.title("mask")

plt.subplot(133)
plt.imshow(y_pred_argmax[random_img])
plt.title("prediction")
plt.show()


################ MeanIoU calculation ##################

num_classes = 4
IOU_keras = MeanIoU(num_classes=num_classes)  
IOU_keras.update_state(y_pred_argmax,y_test_argmax)
print("Mean IoU =", IOU_keras.result().numpy())




