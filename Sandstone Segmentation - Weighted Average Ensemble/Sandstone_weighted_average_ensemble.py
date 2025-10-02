"""
                 (Sandstone segmentation - Ensemble Learning)

"""

###################### Libraries ######################

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv 
from sklearn.model_selection import train_test_split 
from keras.models import load_model 
from keras.utils import to_categorical
from tqdm import tqdm
import segmentation_models as sm
import glob
from keras.metrics import MeanIoU
from sklearn.preprocessing import LabelEncoder 


################## Create a dataset ###################

Data_dir="F:/Python/Datasets/Sandstone/128_patches/"
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


##################### Label Encoder ###################

labelencoder = LabelEncoder()
training_mask=labelencoder.fit_transform(mask_dataset.reshape(-1,1))
training_mask=training_mask.reshape(mask_dataset.shape)


################### Split the data ####################

x_train, x_test, y_train, y_test = train_test_split(image_dataset , training_mask , 
                                                    test_size=0.1 , random_state=49)


################# Data preprocessing ##################

y_train=to_categorical(y_train , num_classes=4)
y_test=to_categorical(y_test , num_classes=4)


#######################################################
############## Weighted average ensemble ##############
"""
    Model1 = U-net with efficientnetb0 backbone
    Model2 = U-net with resnet34 backbone
    Model3 = U-net with vgg16 backbone
    
"""

model1=load_model("F:/Python/CNN/U-Net/Sandstone/effbo/sandstone-effb0--40--0.9255310893058777.hdf5",compile=False)
model2=load_model("F:/Python/CNN/U-Net/Sandstone/resnet34/sandstone-res34--50--0.9279112815856934.hdf5",compile=False)
model3=load_model("F:/Python/CNN/U-Net/Sandstone/vgg16/sandstone-vgg16--48--0.9380998611450195.hdf5",compile=False)

backbone1="efficientnetb0"
backbone2="resnet34"
backbone3="vgg16"

input_process1=sm.get_preprocessing(backbone1)
input_process2=sm.get_preprocessing(backbone2)
input_process3=sm.get_preprocessing(backbone3)

x_test1=input_process1(x_test)
x_test2=input_process2(x_test)
x_test3=input_process3(x_test)

pred1=model1.predict(x_test1)
pred2=model2.predict(x_test2)
pred3=model3.predict(x_test3)

y_test_argmax=np.argmax(y_test , axis=3)
y_pred1_argmax=np.argmax(pred1, axis=3)
y_pred2_argmax=np.argmax(pred2, axis=3)
y_pred3_argmax=np.argmax(pred3, axis=3)

preds=[pred1 , pred2 , pred3]
preds=np.array(preds)

weights=[0.3 , 0.1 , 0.6]

weights_pred=np.tensordot(preds , weights , axes=((0),(0)))

weights_pred_argmax=np.argmax(weights_pred , axis=3)


n_classes = 4
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes) 


IOU1.update_state(y_test_argmax, y_pred1_argmax)
IOU2.update_state(y_test_argmax, y_pred2_argmax)
IOU3.update_state(y_test_argmax, y_pred3_argmax)
IOU_weighted.update_state(y_test_argmax ,weights_pred_argmax)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())


################# Finding best weights ################

n_classes = 4
MeanIoU_list=[]
for w1 in range(0,10):
    for w2 in range(0,10):
        for w3 in range(0,10):
            wts = [w1/10., w2/10., w3/10.]
            if sum(wts) == 1:
                ensembled_preds = np.tensordot(preds, wts, axes=((0),(0)))
                ensembled_pred_argmax = np.argmax(ensembled_preds, axis=3)
                IOU = MeanIoU(num_classes=n_classes)  
                IOU.update_state(y_test_argmax, ensembled_pred_argmax)
                mean_iou=IOU.result().numpy()
                print(mean_iou , wts) 
                MeanIoU_list.append([mean_iou , wts])
            
print("maximum MeanIoU you can get is :{} ".format(max(MeanIoU_list)))


############## Ensemble model evaluation ##############

weights=[0.3 , 0.3 , 0.4]

weights_pred=np.tensordot(preds, weights, axes=((0),(0)))

weights_pred_argmax=np.argmax(weights_pred, axis=3)

y_pred1_argmax=np.argmax(pred1, axis=3)
y_pred2_argmax=np.argmax(pred2, axis=3)
y_pred3_argmax=np.argmax(pred3, axis=3)


n_classes = 4
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes) 


IOU1.update_state(y_test_argmax, y_pred1_argmax)
IOU2.update_state(y_test_argmax, y_pred2_argmax)
IOU3.update_state(y_test_argmax, y_pred3_argmax)
IOU_weighted.update_state(y_test_argmax ,weights_pred_argmax)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())


######## Plot predictions from ensemble model #########

rand_number=np.random.randint(0,x_test.shape[0])

def mean_iou_counter(model_argmax , test_mask_argmax , img_num , num_classes=4):
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(model_argmax[img_num], test_mask_argmax[img_num])
    return IOU_keras.result().numpy()

MeanIoU_effb0=(mean_iou_counter(y_pred1_argmax,y_test_argmax,rand_number))*100
MeanIoU_resnet34=(mean_iou_counter(y_pred2_argmax,y_test_argmax,rand_number))*100
MeanIoU_vgg16=(mean_iou_counter(y_pred3_argmax,y_test_argmax,rand_number))*100
MeanIoU_ensembled=(mean_iou_counter(weights_pred_argmax,y_test_argmax,rand_number))*100

fig,ax=plt.subplots(4,3,figsize=(12,12), dpi=200, facecolor="black")
ax[0,0].imshow(x_test[rand_number])
ax[0,0].set_title("test image", color="white", fontsize=15)

ax[0,1].imshow(y_test_argmax[rand_number])
ax[0,1].set_title("test mask", color="white", fontsize=15)

ax[0,2].imshow(y_pred1_argmax[rand_number])
ax[0,2].set_title("effb0 prediction", color="white", fontsize=15)
ax[0,2].set_ylabel(f"MeanIoU:{MeanIoU_effb0:.2f}" , color="white", fontsize=15)

ax[1,0].imshow(x_test[rand_number])
ax[1,0].set_title("test image", color="white", fontsize=15)

ax[1,1].imshow(y_test_argmax[rand_number])
ax[1,1].set_title("test mask", color="white", fontsize=15)

ax[1,2].imshow(y_pred2_argmax[rand_number])
ax[1,2].set_title("resnet34 prediction", color="white", fontsize=15)
ax[1,2].set_ylabel(f"MeanIoU:{MeanIoU_resnet34:.2f}" , color="white", fontsize=15)

ax[2,0].imshow(x_test[rand_number])
ax[2,0].set_title("test image", color="white", fontsize=15)

ax[2,1].imshow(y_test_argmax[rand_number])
ax[2,1].set_title("test mask", color="white", fontsize=15)

ax[2,2].imshow(y_pred3_argmax[rand_number])
ax[2,2].set_title("vgg16 prediction", color="white", fontsize=15)
ax[2,2].set_ylabel(f"MeanIoU:{MeanIoU_vgg16:.2f}" , color="white", fontsize=15)

ax[3,0].imshow(x_test[rand_number])
ax[3,0].set_title("test image", color="white", fontsize=15)

ax[3,1].imshow(y_test_argmax[rand_number])
ax[3,1].set_title("test mask", color="white", fontsize=15)

ax[3,2].imshow(weights_pred_argmax[rand_number])
ax[3,2].set_title("ensemble prediction", color="white", fontsize=15)
ax[3,2].set_ylabel(f"MeanIoU:{MeanIoU_ensembled:.2f}" , color="white", fontsize=15)

plt.show()










