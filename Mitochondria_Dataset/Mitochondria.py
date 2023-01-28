from pyexpat import model
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 as cv 
import os 
from PIL import Image 
from keras.utils import normalize
from unet_model import unet_model
import random


img_dir="F:/Python/mitocondery/training/img"
mask_dir="F:/Python/mitocondery/training/msk"

image_dataset=[]
mask_dataset=[]

images=os.listdir(img_dir)
for i,image_name in enumerate(images):
    if image_name.split(".")[1]=="tif":
        image=cv.imread(img_dir+"/"+image_name,0)
        image=Image.fromarray(image)
        image=image.resize((256,256))
        image_dataset.append(np.array(image))


masks=os.listdir(mask_dir)
for i,mask_name in enumerate(masks):
    if mask_name.split(".")[1]=="tif":
        mask=cv.imread(mask_dir+"/"+mask_name,0)
        mask=Image.fromarray(mask)
        mask=mask.resize((256,256))
        mask_dataset.append(np.array(mask)) 

image_dataset=np.expand_dims(normalize(np.array(image_dataset),axis=1),3)
mask_dataset=np.expand_dims(np.array(mask_dataset),3)/255.0


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(image_dataset,mask_dataset,test_size=0.1,random_state=0)

#sanity check
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(x_train[100],cmap="gray")
plt.subplot(122)
plt.imshow(y_train[100] , cmap="gray")
plt.show()
#%%
def get_model():
    return unet_model(256,256,1)

model=get_model()
history=model.fit(x_train,y_train,batch_size=16,epochs=50,validation_data=(x_test,y_test),shuffle=False)
model.save("mito_model.hdf5")



# %% IOU metric 
y_pred=model.predict(x_test)
y_pred_threshold=y_pred>0.5

intersection=np.logical_and(y_test,y_pred_threshold)
union=np.logical_or(y_test,y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE : {}".format(IOU))
# print(y_pred)

# %%
model = get_model()
model.load_weights('mito_model.hdf5')

test_img_number = random.randint(0, len(x_test))
test_img = x_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

test_img_other = cv.imread('data/test_images/02-1_256.tif', 0)
#test_img_other = cv2.imread('data/test_images/img8.tif', 0)
test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()
