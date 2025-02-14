#%%
import skimage.morphology as morpho

import numpy as np
import matplotlib.pyplot as plt

# Dimensions of the image
height = 100
width = 100

# Create a 2D array for the image
image = np.ones((height, width))

# Define the center column and calculate the gradient
center_col = width // 2
for row in range(height):
    for col in range(width):
        distance_from_center = 20*abs(col - center_col)
        # Normalize the distance and set pixel value
        image[row, col] = distance_from_center / (center_col)

two = 21
one = 20
three = 22
image[two, two]=0
image[three, three]=0
image[two, one]=0
image[one, two]=0
image[one, one]=0
image[three, one]=0
image[one, three]=0
image[two, three]=0
image[three, two]=0


# Clip values to be between 0 and 1 for display purposes
image = np.clip(image, 0, 1)

# Plot the image
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray', origin='upper')
plt.axis('off')
plt.title("Gradient Image with Center Column = 0")
plt.show()


#%%
element = morpho.rectangle(1, 3)
plt.imshow(element, cmap='gray', origin='upper')
plt.show()
new_img = morpho.closing(image, element)
plt.imshow(new_img, cmap='gray', origin='upper')
plt.show()

#%%
element = morpho.rectangle(8, 1)
new_img = morpho.closing(image, element)
plt.imshow(new_img, cmap='gray', origin='upper')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Dimensions of the image
height = 100
width = 100

# Create a 2D array for the image
image = np.ones((height, width))

# Define the center column and calculate the gradient
center_col = width // 2
for row in range(height):
    for col in range(width):
        distance_from_center = abs(col - center_col)
        # Normalize the distance and set pixel value
        image[row, col] = distance_from_center / (center_col)

# Clip values to be between 0 and 1 for display purposes
image = np.clip(image, 0, 1)

# Add random noise
noise = np.random.random((height, width)) * 0.2  # Scale noise by 0.2
noisy_image = image + noise

# Clip values to ensure they remain between 0 and 1
noisy_image = np.clip(noisy_image, 0, 1)

# Plot the noisy image
plt.figure(figsize=(6, 6))
plt.imshow(noisy_image, cmap='gray', origin='upper')
plt.axis('off')
plt.title("Noisy Gradient Image")
plt.show()

#%%
element = morpho.rectangle(1, 8)
plt.imshow(element, cmap='gray', origin='upper')
plt.show()
new_img = morpho.closing(image, element)
plt.imshow(new_img, cmap='gray', origin='upper')
plt.show()

