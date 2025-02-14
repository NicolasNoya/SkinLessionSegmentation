#%%
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Sample image as a numpy array (grayscale)
image = np.random.rand(100, 100)

# Zoom factors (scale the image by 0.5 in both dimensions)
zoom_factors = (0.5, 0.5)

# Resize the image
resized_image = zoom(image, zoom_factors)

# Display the original and resized images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")

ax[1].imshow(resized_image, cmap='gray')
ax[1].set_title("Resized Image")

plt.show()

#%%
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Sample 3-channel RGB image as a numpy array (100x100 with 3 channels)
image = np.random.rand(100, 100, 3)

# Zoom factors (scale the image by 0.5 in height and width, leave the channels intact)
zoom_factors = (0.5, 0.5, 1)  # Only resize the height and width

# Resize the image
resized_image = zoom(image, zoom_factors)

# Display the original and resized images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Original Image")

ax[1].imshow(resized_image)
ax[1].set_title("Resized Image")

plt.show()
