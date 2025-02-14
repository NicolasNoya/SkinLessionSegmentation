#%%
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# Example binary mask (replace with your own mask)
mask = np.array([[0, 0, 1, 1, 0],
                 [0, 1, 1, 0, 0],
                 [1, 1, 0, 0, 1],
                 [0, 0, 0, 1, 1]])

plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()

# Label the connected components
labeled_mask, num_labels = measure.label(mask, background=0, return_num=True)

# Generate individual masks for each object
individual_masks = [(labeled_mask == i).astype(np.uint8) for i in range(1, num_labels + 1)]

# Visualize the individual masks
fig, ax = plt.subplots(1, num_labels, figsize=(12, 4))
for i, individual_mask in enumerate(individual_masks):
    ax[i].imshow(individual_mask, cmap='gray')
    ax[i].set_title(f'Object {i+1}')
    ax[i].axis('off')
plt.show()