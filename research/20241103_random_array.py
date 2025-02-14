#%%
from scipy.ndimage import zoom
import os
import sys
import numpy as np
sys.path.append("../")
import matplotlib.image as mpimg
import manager
from manager import Manager
import matplotlib.pyplot as plt
import cv2 

#%%
channel = mpimg.imread("../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000030.jpg")
grey_hist = np.histogram(channel[0], bins=255, range=(0,255))
channel.shape

plt.imshow(channel)
plt.show()

random_array = np.random.uniform(0, 255, (channel.shape[0], channel.shape[1]))
plt.imshow(random_array, cmap='gray')
plt.show()

# %%
