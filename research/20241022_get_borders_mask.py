#%%
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%%

# Create a simple binary mask (e.g., a white square in a black background)
mask = np.zeros((110, 100), dtype=np.uint8)
mask[30:70, 30:70] = 255  # A white square in the middle
print((mask))
#%%

mask = mpimg.imread("../dataset/skin_lesion_dataset-master/prueba/ISIC_0000030_Segmentation.png").astype(np.uint8)
original_image = mpimg.imread("../dataset/skin_lesion_dataset-master/prueba/ISIC_0000030.jpg")/255

print((mask))
plt.imshow(mask, cmap='gray')
plt.show()


#%%
# Method 1: Using findContours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#%%

# Create an empty image to draw the contours
contour_image = np.zeros_like(mask)
cv2.drawContours(original_image, contours, -1, (255), 1)  # Draw contours

#%%
# Method 2: Using Canny edge detection
edges = cv2.Canny(mask, 0, 1)
transposed = []
# transposed.append(original_image.transpose()[0] - edges.transpose())
transposed.append(original_image.transpose()[0])
transposed.append(original_image.transpose()[1] - edges.transpose())
# transposed.append(original_image.transpose()[1])
transposed.append(original_image.transpose()[2])
# transposed.append(original_image.transpose()[2] - edges.transpose())

#%%
plt.imshow(np.array(transposed).transpose())
plt.show()

#%%
plt.imshow(np.array(edges))
plt.show()
#%%



# Plotting the results
plt.figure(figsize=(10, 5))

# Original Mask
plt.subplot(1, 3, 1)
plt.imshow(mask, cmap='gray')
plt.title("Original Mask")
plt.axis('off')


# Edges using Canny edge detection
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Borders (Canny)")
plt.axis('off')

plt.tight_layout()
plt.show()

# %%
