#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


#%%

def channel_extractor(im):


    X_channel = extract_channel_X(im)
    XoYoR_channel = extract_channel_XoYoR(im)
    XoYoZoR_channel = extract_channel_XoYoZoR(im)
    R_channel = extract_channel_R(im)

    # Convert to tensor 
    X_ten = torch.tensor(X_channel).float()
    XoYoR_ten = torch.tensor(XoYoR_channel).float()
    XoYoZoR_ten = torch.tensor(XoYoZoR_channel).float()
    R_ten = torch.tensor(R_channel).float()

    return [X_ten, XoYoR_ten, XoYoZoR_ten, R_ten]
#%%
#define an extractor that, given an image, convert to a color space and extract the desired channels
#our images are in RGB
def extract_channel_X(im):
    X_channel = im[:,:,0]
    return X_channel
#%%

def extract_channel_R(im):
    R_channel = im[:,:,0]
    return R_channel
#%%

T = torch.tensor([
    [0.4125, 0.3576, 0.1804],
    [0.2127, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9502]
], dtype=torch.float32)

def converter_rgb_to_XoYoZ(im): #without using tensor
    im=im/255
    r,c,d=im.shape
    im_conv = np.zeros((r * c, T.shape[0]))#initialize the image with same shape 
    #i want to reduce the image to two dimensions
    im = im.reshape(r*c,d)
    #multiply by the transformation matrix
    for i in range(r*c):
        
            im_conv[i] = np.matmul(T,im[i])
    #reshape into the original shape
    im_conv = im_conv.reshape(r,c,T.shape[0])
    #rescale 
    im_conv = im_conv*255.0
    return im_conv
# %%

def extract_channel_XoYoR(im):
    im_conv=converter_rgb_to_XoYoZ(im)
    im=im/255.0
    im_conv = im_conv/255.0

    XoYoR_channel = im_conv[:,:,1]+ im_conv[:,:,2]+im[:,:,0]
    #rescale
    XoYoR_channel = XoYoR_channel/3*255.0
    return XoYoR_channel

#%%
def extract_channel_XoYoZoR(im):
    im_conv=converter_rgb_to_XoYoZ(im)
   
    #normalize the image
    im=im/255.0
    im_conv = im_conv/255.0
    #extract
    XoYoZoR_channel = im_conv[:,:,0]+im_conv[:,:,1]+im_conv[:,:,2]+im[:,:,0]

    XoYoZoR_channel = XoYoZoR_channel/4*255.0
    return XoYoZoR_channel
   
   

# %%
#i want to show the ISIC image without skio library

#load the image
im_path= '/Users/cecilia/Desktop/ Examples/ISIC_0000030.jpg'
im = Image.open(im_path)

im = np.array(im)
#convert to tensor
im_tensor = torch.tensor(im).float()
#show the image
plt.imshow(im_tensor/255.0)
plt.show()
#i want to plot the result of the channel extractor
X_channel, XoYoR_channel, XoYoZoR_channel, R_channel=channel_extractor(im)
plt.imshow(XoYoZoR_channel)
plt.show()
plt.imshow(XoYoR_channel)
plt.show()
plt.imshow(X_channel)
plt.show()
plt.imshow(R_channel, cmap='gray')
plt.show()
#since they seem pretty similar, i will try to plot the difference

#diff = X_channel-XoYoZoR_channel
#plt.imshow(diff)
#plt.show()
#check it numerically
#print(torch.max(diff))
#print(torch.min(diff))


# %%
#MEDIAN FILTER
#I want to preprocess the images with a median filter, in order to remove the noise
#I need to consider the fact I have RGB images, so I need to separate the channels and apply the filter to each of them
def check_median_filter(im, typ=1, r=20, xy=None):
    # Check if we have an RGB with 3 dimensions
    if im.ndim == 3:
        # Extract the three channels 
        r_channel = im[:, :, 0]  
        g_channel = im[:, :, 1]  
        b_channel = im[:, :, 2]  
       
        r_filtered = median_filter(r_channel, typ, r, xy)  

       
        g_filtered = median_filter(g_channel, typ, r, xy)  

        
        b_filtered = median_filter(b_channel, typ, r, xy)  

        #recombine the three channels
        im_filtered = np.stack([r_filtered, g_filtered, b_filtered], axis=-1)
        return im_filtered
    else:
        #if the image has only two dimensions, apply the filter
        return median_filter(im, typ, r, xy)


def median_filter(im, typ=1, r=20, xy=None):
    
    lx = []
    ly = []
    # Assicurati che im sia bidimensionale (altezza, larghezza)
    if im.ndim != 2:
        raise ValueError("Median_filter works only with bidimensional images")
    
    (ty, tx) = im.shape  
    
    if typ == 1:  #If typ=1, create a square window
        for k in range(-r, r + 1):
            for l in range(-r, r + 1):
                lx.append(k)
                ly.append(l)
    elif typ == 2:  # if typ=2, create a circular window
        for k in range(-r, r + 1):
            for l in range(-r, r + 1):
                if k**2 + l**2 <= r**2:
                    lx.append(k)
                    ly.append(l)
    else:  # window created by the user
        lx, ly = xy

    # compute the limits 
    debx = -min(lx)
    deby = -min(ly)
    finx = tx - max(lx)
    finy = ty - max(ly)
    ttx = finx - debx
    tty = finy - deby

    # matrix to store the values of the pixels
    tab = np.zeros((len(lx), ttx * tty))

    # fill the matrix with the values of the pixels
    for k in range(len(lx)):
        tab[k, :] = im[deby + ly[k]:deby + tty + ly[k], debx + lx[k]:debx + ttx + lx[k]].reshape(-1)

    # apply the median filter
    out = im.copy()
    out[deby:finy, debx:finx] = np.median(tab, axis=0).reshape((tty, ttx))

    return out


# %%
#Trying to apply the functions to the image
#load the image
im_path= '/Users/cecilia/Desktop/ Examples/ISIC_0000030.jpg'
im = Image.open(im_path)

im = np.array(im)
#convert to tensor
im_tensor = torch.tensor(im).float()
#show the image
plt.imshow(im_tensor/255.0)
plt.show()
im=check_median_filter(im,typ=2,r=20)
#i want to plot the result of the channel extractor
X_channel, XoYoR_channel, XoYoZoR_channel, R_channel=channel_extractor(im)
plt.imshow(XoYoZoR_channel)
plt.show()
plt.imshow(XoYoR_channel)
plt.show()
plt.imshow(X_channel)
plt.show()
plt.imshow(R_channel, cmap='gray')
plt.show()

# %%









