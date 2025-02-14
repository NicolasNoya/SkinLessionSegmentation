#%%
import sys
sys.path.append('../segmentation')
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from segmentation import Segmentator
import os
#%%

segmentator = Segmentator()
for i in os.listdir('../dataset/skin_lesion_dataset-master/melanoma'):
    path = f'../dataset/skin_lesion_dataset-master/melanoma/{i}'
    if "Segmentation" in i:
        continue
    img = mpimg.imread(path)/255
    final_image = segmentator.whole_stack(img, 2, 30)

    original_image = mpimg.imread(path)
    plt.imshow(original_image)
    plt.show()

    plt.imshow(final_image.transpose(), cmap='gray')
    plt.show()

#%%
segmentator = Segmentator()
path = f'../dataset/ISIC_0000030.jpg'
img = mpimg.imread(path)/255
plt.imshow(img)
plt.show()
#%%
transpose_img = img.transpose()
corners_mask = segmentator.preprocess_corners(transpose_img, (img.shape[1]+img.shape[0])/4.2)
plt.imshow(corners_mask.transpose())
plt.show()
#%%
masked_image_2 = segmentator.apply_a_mask(img, corners_mask)
plt.imshow(masked_image_2.astype(np.float32).transpose())
plt.show()
#%%
threshold = segmentator.otsus_function(corners_mask)
plt.imshow(threshold.transpose(), cmap='gray')
plt.show()
# threshold
#%%
segmentator = Segmentator()
def otsus_function(image: np.ndarray)->np.ndarray:
    """
    This functions returns the threshold value for the Otsu's method. 

    Input:
    - image: The image to be thresholded, this image should 
        have only one channel.
    Output:
    - mask: The mask with the pixels above the threshold in white and the rest.
    """
    if image.shape[0] > 5 and len(image.shape) > 2:
        image = image.transpose()
    masks = []
    thresholds = []
    for channel in image:
        # change a little the rgb conversion to top the brightness to 255
        grey_hist = np.histogram(channel, bins=255, range=(0, 1))
        uzip = list(zip(grey_hist[0], (grey_hist[1]).astype(np.float32)))
        print("The zipped histogram is: ",uzip)
        new_sigma = -1
        new_threshold = 0
        for t in range(1, 253):# TODO: This could be a problem when the image is normalized
            w0 = sum(grey_hist[0][:t]) 
            w1 = sum(grey_hist[0][t:]) 
            if w0 == 0 or w1 == 0:
                continue
            media_0_t = 1/w0 * sum(uzip[i][1] * uzip[i][0] for i in range(0, t))
            media_1_t = 1/w1 * sum(uzip[i][1] * uzip[i][0] for i in range(t+1, 255))
            sigma_0_t = 1/w0 * sum((uzip[i][1] - media_0_t)**2 * uzip[i][0] for i in range(0, t))
            sigma_1_t = 1/w1 * sum((uzip[i][1] - media_1_t)**2 * uzip[i][0] for i in range(t+1, 255))
            sigma_b = ((w0)/(w0+w1) * sigma_0_t + (w1)/(w0+w1) * sigma_1_t)

            # Get the minimum sigmab
            if sigma_b < new_sigma or new_sigma == -1:
                new_threshold = grey_hist[1][t] 
                new_sigma = sigma_b
        print("The new threshold is: ",new_threshold)

        thresholds.append(new_threshold)
    
    for index, threshold in enumerate(thresholds):
        masks.append(segmentator.apply_a_threshold(image[index], threshold))
                
    sum_mask = np.zeros(masks[0].shape)
    for channel in masks:
        sum_mask += channel
    final_mask = np.where(sum_mask>0, 1, sum_mask)

    return final_mask

threshold = otsus_function(corners_mask)
plt.imshow(threshold.astype(np.float32).transpose(), cmap='gray')
plt.show()




# %%
import numpy as np
import matplotlib.pyplot as plt

def crear_mascara_circular(tamaño, radio):
    """
    Crea una máscara circular con valores de 1 dentro de un radio 'r' desde el centro.
    
    Parámetros:
    - tamaño: El tamaño de la máscara (una tupla, como (altura, anchura)).
    - radio: El radio del círculo.
    
    Retorna:
    - Una máscara 2D con 1 dentro del círculo y 0 fuera.
    """
    # Crear un grid de coordenadas
    h, w = tamaño
    Y, X = np.ogrid[:h, :w]

    # Calcular las distancias desde el centro
    centro_y, centro_x = h // 2, w // 2
    distancias_al_centro = (X - centro_x)**2 + (Y - centro_y)**2

    # Crear la máscara: 1 dentro del radio, 0 fuera
    mascara = distancias_al_centro <= radio**2
    
    return mascara.astype(int)  # Convertir a entero para tener 0 y 1

# Ejemplo de uso
tamaño = (200, 200)  # Tamaño de la máscara (200x200 píxeles)
radio = 50  # Radio del círculo

mascara = crear_mascara_circular(tamaño, radio)

# Visualización de la máscara
plt.imshow(mascara, cmap='gray')
plt.title('Máscara Circular')
plt.axis('off')
plt.show()

#%%
segmentator = Segmentator()
path = f'../dataset/ISIC_0000030.jpg'
final_image = segmentator.whole_stack(path)

original_image = mpimg.imread(path)

plt.imshow(original_image)
plt.show()

plt.imshow(final_image.transpose())
plt.show()


#%%

img = mpimg.imread(path)/255
contrast_factor = 2
mean = img.mean()

image_contrasted = contrast_factor * (img - mean) + mean
plt.imshow(image_contrasted)
plt.show()
#%%
path = "../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000046.jpg"
img =  mpimg.imread(path)/255

plt.imshow(img, cmap='gray')
plt.show()

segmentator = Segmentator()
final_image = segmentator.whole_stack(img, contrast_factor=2, maximum_filter_size=40)
plt.imshow(final_image.transpose(), cmap='gray')
plt.show()


#%%
print(img.transpose()[2].shape)
#%%
path = f'../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000046.jpg'
img = mpimg.imread(path)/255
final_image = segmentator.whole_stack(img, 2, 10)

original_image = mpimg.imread(path)
plt.imshow(original_image)
plt.show()

plt.imshow(final_image.transpose(), cmap='gray')
plt.show()


# %%
