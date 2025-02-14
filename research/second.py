# %%creating a circular filter to remove noise
from scipy.ndimage import convolve
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%%

def circular_avg_filter(radius):
    '''This function creates a circular filter of a given radius, creating a mask of 1 inside the circle and 0 outside'''
    h,w=np.ogrid[-radius:radius+1,-radius:radius+1] 
    mask = h**2+w**2<=radius**2
    mask_size=radius*2+1 
    #create an empty matrix to store the values
    kernel_val=np.zeros((mask_size,mask_size),dtype=float)
    #fill the matrix with 1 inside the circle
    kernel_val[mask]=1
    #I normalize so I won't change the intensity of the image. The sum of the values of the kernel is 1
    kernel_val=kernel_val/np.sum(kernel_val)
    return kernel_val
#%%
def apply_circular_avg_filter(radius,im): 
    '''This fuction apply the filter: the filter will be apply to each pixel, to each channel'''
    im_array = np.array(im)#filters are applied to arrays so I  convert to an array of type uint8
    filter=circular_avg_filter(radius)
    #apply the filter using the convolve function
    filtered_image = np.stack([convolve(im_array[:,:,i], filter, mode='reflect') for i in range(3)], axis=-1)
    #convert from an array to an image of type uint8
    filtered_image=Image.fromarray(filtered_image)
    return filtered_image 

#%%
radius=5
path='../dataset/ISIC_0000030.jpg'
im = Image.open(path)
# Plot image original
plt.imshow(im)
plt.show()
#apply the filter
filtered_image=apply_circular_avg_filter(radius,im)
#plot the filtered image
#the image was an array, so the filtered image is an array
#I need to convert it to an image of type uint8
filtered_image = np.array(filtered_image, dtype=np.uint8)
# Plot image filtered
plt.imshow(filtered_image)
plt.show()
# %%
