#%%
from scipy.ndimage import zoom
import os
import sys
import numpy as np
sys.path.append("../")
import matplotlib.image as mpimg
from manager import Manager
import matplotlib.pyplot as plt
import cv2 
#%%
#
man = Manager(image_path= "../dataset/skin_lesion_dataset-master/prueba/")

man.full_stack()
#%%
plt.imshow(man.X_channel, cmap='grey')
plt.show()
plt.imshow(man.XoYoR_channel, cmap='grey')
plt.show()
plt.imshow(man.XoYoZoR_channel, cmap='grey')
plt.show()
plt.imshow(man.R_channel, cmap='grey')
plt.show()


#%%
plt.hist(man.X_channel)
plt.show()

#%%
mask = np.where(man.X_channel>0.55, 0, 1)
plt.imshow(mask.transpose())
plt.show()