############################ Libraries ############################

import numpy as np 
import matplotlib.pyplot as plt 
import tifffile as tif
from tqdm import tqdm

############################ Read Files ###########################

path=".../path/to/dataset/folder/"
image_dataset=tif.imread(path + "images_as_128x128_patches.tif")
mask_dataset=tif.imread(path + "masks_as_128x128_patches.tif")

########################### Random Plot ###########################

rand_num=np.random.randint(0,1600)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(image_dataset[rand_num])
plt.title("original image")
plt.subplot(122)
plt.imshow(mask_dataset[rand_num])
plt.title("original mask")
plt.show()

################### Save Each Pach in Directory ###################

save_path=".../path/to/save/directory/"
for i in tqdm(range(image_dataset.shape[0])):
    image=image_dataset[i]
    tif.imwrite(save_path + "image/image_" + str(i) + ".jpg", image)

for j in tqdm(range(mask_dataset.shape[0])):
    mask=mask_dataset[j]
    tif.imwrite(save_path + "mask/mask_" + str(j) + ".jpg", mask)
