#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
#%%
img = mpimg.imread('../dataset/ISIC_0000030.jpg')
print(img.transpose().shape)
print(img.transpose())
transpose_img = img.transpose()
#%%
plt.imshow(img)
#%%
histogram = plt.hist(transpose_img[2])
plt.show(histogram)
#%%
numpy_hist = np.histogram(transpose_img[2], bins=256, range=(0, 256))
print((numpy_hist[1]))
#%%
height, width = 300, 300
image = np.zeros((height, width),dtype=np.uint8)

center = (int(height/2), int(width/2))
radius = 100
y, x = np.ogrid[:height, :width]
dist_from_center = np.sqrt((x-center[1]**2+(y-center[0])**2))
mask = dist_from_center <= radius
int_mask = mask.astype(np.uint8)
print(int_mask)
masked_image = cv2.bitwise_and(image,image, mask=int_mask)
# image[mask] = 255
# print(image[center[0]][center[1]+1])
#%%
print(masked_image)
# plt.show(masked_image)


#%%

plt.show()
#%%
cv2.imshow('image', img)
#%%
x = np.random.normal(170, 10, 250)
#%%
print(len(x))
plt.hist(x)
plt.show()
#%%
plt.imshow(x)

#%%
print(x)

#%%
masked_random = np.where(x>130, x, 0)
print(masked_random)
#%%
print((masked_random).shape)
#%%
plt.show(masked_random)

#%%
# Apply a mask to an image
random_image = np.array(np.random.randint(0, 256, (255, 255), dtype=np.uint8))
masked_img = np.where(random_image>130, 255.0, 0.0)

img = Image.fromarray(masked_img)
plt.imshow(img)
#%%
# Detect the black corner/borders
img = mpimg.imread('../dataset/ISIC_0000030.jpg')

transpose_img = img.transpose()
plt.imshow(img)
#%%
transpose_img.shape
#%%

borders_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

border_length = 130

borders_mask[:border_length, : ] = 1
borders_mask[-border_length:, :] = 1
borders_mask[:, :border_length] = 1
borders_mask[:, -border_length:] = 1

r_black = np.where(transpose_img[0]>=50, 0, 255)
g_black = np.where(transpose_img[1]>=50, 0, 255)
b_black = np.where(transpose_img[2]>=50, 0, 255)

total = (r_black + g_black + b_black)/3 * borders_mask.transpose()
print(total[int(total.shape[0]/2)][int(total.shape[1]/2)])
print(total)

rgb_black = np.where(total>=175, 255, 0)

plt.imshow(rgb_black.transpose())
#%%
histogram = plt.hist(rgb_black.transpose())
plt.show(histogram)
#%%
rgb_black # Border's mask
#%%
# Otsus algorithm

img = mpimg.imread('../dataset/ISIC_0000030.jpg')
transpose_img = img.transpose()
histogram = np.histogram(transpose_img[2], bins=1, range=(0,0))

#%%
sum_rgb = (0.293 * transpose_img[0] + 0.574 * transpose_img[1] + 0.133 *transpose_img[2]) 
# small change in the rgb->gray conversion formula to top the brightness to 255
plt.imshow(sum_rgb.transpose(), cmap='gray')
#%%
plt.imshow(transpose_img[2], cmap='gray')

#%%
plt.imshow(sum_rgb.transpose(), cmap='gray')

print(sum_rgb.shape)
#%%
grey_hist = np.histogram(sum_rgb, bins=256, range=(0, 256))
print((grey_hist[1]))
#%%
print((grey_hist[0][1]))

#%%
thresholds = []
uzip = list(zip(grey_hist[0], (grey_hist[1]).astype(int)))
#%%
for t in range(1, 254):
    w0 = sum(grey_hist[0][:t]) 
    w1 = sum(grey_hist[0][t:]) 
    if w0 == 0 or w1 == 0:
        continue
    media_0_t = 1/w0 * sum(uzip[i][1] * uzip[i][0] for i in range(0, t))
    media_1_t = 1/w1 * sum(uzip[i][1] * uzip[i][0] for i in range(t+1, 255))
    # print(media_0_t, media_1_t)
    sigma_0_t = 1/w0 * sum((uzip[i][1] - media_0_t)**2 * uzip[i][0] for i in range(0, t))
    sigma_1_t = 1/w1 * sum((uzip[i][1] - media_1_t)**2 * uzip[i][0] for i in range(t+1, 255))
    print(sigma_0_t, sigma_1_t)
    sigma_b = ((w0)/(w0+w1) * sigma_0_t + (w1)/(w0+w1) * sigma_1_t)
    print(sigma_b)
    # thresholds.append((((w0)/(w0+w1) * sigma_0_t + (w1)/(w0+w1) * sigma_1_t), t))
    thresholds.append((sigma_b, t))
    # thresholds.append((((1 * sigma_0_t + 1 * sigma_1_t), t)))

#%%

final_threshold = min(thresholds)[1]
print(final_threshold)
#%%
final_mask = np.where(sum_rgb>final_threshold, 255, 0)
plt.imshow(final_mask, cmap='gray')


#%%

# Python program to illustrate 
# Otsu thresholding type on an image 
  
# organizing imports 
import cv2          
import numpy as np     
  
# path to input image is specified and 
# image is loaded with imread command 
image1 = cv2.imread('./dataset/ISIC_0000030.jpg') 
  
# cv2.cvtColor is applied over the 
# image input with applied parameters 
# to convert the image in grayscale 
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
  
# applying Otsu thresholding 
# as an extra flag in binary  
# thresholding      
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)      
  
# the window showing output image          
# with the corresponding thresholding          
# techniques applied to the input image     
cv2.imshow('Otsu Threshold', thresh1)  



#%%
print(uzip[0][1] - uzip[0][0])
#%%
# Convert the image to a NumPy array (RGB format)
img_array = np.array(img)

# Convert the RGB image to grayscale
# The weights [0.2989, 0.5870, 0.1140] correspond to the RGB to grayscale conversion formula
gray_img = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])

# Display the grayscale image using matplotlib
plt.imshow(gray_img, cmap='gray')
plt.show()
#%%

import segmentation.segmentation as segmentation
segmentator = segmentation.Segmentator()
img = mpimg.imread('./dataset/ISIC_0000030.jpg')
transpose_img = img.transpose()
corners_mask = segmentator.preprocess_corners(transpose_img)
masked_image = segmentator.apply_a_mask(img, corners_mask)
plt.imshow(masked_image.transpose())


#%%
list_a = [1,2,3,4,5,6]
print(list_a[:-1])