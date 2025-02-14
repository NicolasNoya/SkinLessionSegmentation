#%%
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



#%%
#I want to define some methods usefull for hair removal algorithms
#i define the method for morpolohical closing with parameter the image in GRAYSCALE and the kernel
#The method consist of two steps: first the dilation and then the erosion
def grayscale_morphological_close(im:np.ndarray, kernel:np.ndarray)->np.ndarray:
    '''This function applies the morphological closing to a grayscale image'''
    dilated_im=cv2.dilate(im, kernel, iterations=1)
    eroded_im=cv2.erode(dilated_im, kernel, iterations=1)
    return eroded_im

#rotated ellipse

def create_rotated_ellipse(kernel_size, angle=45):
    """
    Crea un elemento strutturale ellittico ruotato di un certo angolo.
    kernel_size: La dimensione del kernel (es. (12, 12)).
    angle: L'angolo di rotazione in gradi (es. 45 per una diagonale).
    """
    # Crea un'immagine vuota per il kernel
    ellipse_img = np.zeros((kernel_size[0], kernel_size[1]), dtype=np.uint8)

    # Disegna un'ellisse piena al centro dell'immagine
    center = (kernel_size[0] // 2, kernel_size[1] // 2)
    axes = (kernel_size[0] // 2, kernel_size[1] // 2)
    cv2.ellipse(ellipse_img, center, axes, angle, 0, 360, 1, -1)

    return ellipse_img


#%%
#We are going to apply the grayscale morphological closing to each channel of the image, using some structure elements for the directions
def hair_identification_second_v (im: np.ndarray, kernel_size=(10, 10)) -> np.ndarray:
    '''Applica il closing morfologico a ciascun canale dell'immagine RGB utilizzando elementi strutturali in tre direzioni.'''
    
    # Divide l'immagine nei tre canali R, G e B
    B_channel = im[:, :, 0]
    G_channel = im[:, :, 1]
    R_channel = im[:, :, 2]
    
    # Crea gli elementi strutturali per le tre direzioni
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size[0], 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size[1]))
    diagonal_ellipse = create_rotated_ellipse(kernel_size, 45)
    diagonal_ellipse_2 = create_rotated_ellipse(kernel_size, 135)

    # Applica il closing morfologico per ogni canale e direzione
    def apply_closing_for_all_directions(channel):
        horizontal_close = grayscale_morphological_close(channel, horizontal)
        vertical_close = grayscale_morphological_close(channel, vertical)
        diagonal_close = grayscale_morphological_close(channel, diagonal_ellipse)
        diago=grayscale_morphological_close(channel, diagonal_ellipse_2)
        return np.maximum.reduce([horizontal_close, vertical_close, diagonal_close,diago])
    
    # Applica il closing su ciascun canale
    B_max = apply_closing_for_all_directions(B_channel)
    G_max = apply_closing_for_all_directions(G_channel)
    R_max = apply_closing_for_all_directions(R_channel)

    # Calcola la differenza tra l'immagine originale e quella chiusa per ciascun canale
    difference_B = cv2.absdiff(B_channel, B_max)
    difference_G = cv2.absdiff(G_channel, G_max)
    difference_R = cv2.absdiff(R_channel, R_max)

    # Applica una soglia per ottenere maschere binarie per ciascun canale
    _, binary_hair_mask_B = cv2.threshold(difference_B, 30, 1, cv2.THRESH_BINARY)
    _, binary_hair_mask_G = cv2.threshold(difference_G, 30, 1, cv2.THRESH_BINARY)
    _, binary_hair_mask_R = cv2.threshold(difference_R, 30, 1, cv2.THRESH_BINARY)

    # Converti le maschere binarie a np.uint8 per eseguire l'operazione OR
    binary_hair_mask_B = binary_hair_mask_B.astype(np.uint8)
    binary_hair_mask_G = binary_hair_mask_G.astype(np.uint8)
    binary_hair_mask_R = binary_hair_mask_R.astype(np.uint8)

    # Unisci le maschere con un'operazione OR
    combined_mask = binary_hair_mask_B | binary_hair_mask_G | binary_hair_mask_R

    return combined_mask

#%%
#OTHER POSSIBLE CODE WITH INTERPOLATION WITH AVG
def average_value_interpolation(image: np.ndarray, mask: np.ndarray, radius=5) -> np.ndarray:
    """
    Replaces hair pixels in the mask with the average RGB value of the surrounding non-hair pixels.
    """
    max_rows, max_cols = mask.shape
    interpolated_image = image.copy()
    
    # Iterate over each pixel in the mask
    for x in range(max_rows):
        for y in range(max_cols):
            # Only process if the pixel is part of the hair
            if mask[x, y] == 1:
                # Calculate the average value of surrounding non-hair pixels
                surrounding_mean = surrounding_non_hair_mean(image, mask, x, y, radius)
                
                # Replace the hair pixel with the average value
                interpolated_image[x, y] = surrounding_mean.astype(np.uint8)

    return interpolated_image

#%%
def euclidean_distance(x1, y1, x2, y2)->float:
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#%%

def bilinear_interpolation(x, y, x1, y1, I1, x2, y2, I2)->float:
    D12=euclidean_distance(x1, y1, x2, y2)
    D1=euclidean_distance(x, y, x1, y1)
    D2=euclidean_distance(x, y, x2, y2)

    #Using the formula of the paper 
    In = I2 * (D1 / D12) + I1 * (D2 / D12) #intenisity in (x,y) pixel
    return In
#%%

def bilinear_interpolation_rgb(x, y, x1, y1, I1, x2, y2, I2, surrounding_mean=None, max_diff_threshold=50, debug=False) -> np.ndarray:
    """
    Bilinear interpolation for an RGB pixel using two reference pixels.

    Parameters:
    - x, y: coordinates of the pixel to interpolate.
    - x1, y1, I1: coordinates and intensity of the first reference pixel.
    - x2, y2, I2: coordinates and intensity of the second reference pixel.
    - surrounding_mean: mean of surrounding non-hair pixels for normalization (optional).
    - max_diff_threshold: maximum allowed difference between I1 and I2 before favoring the surrounding mean.
    - debug: flag for printing debug information.

    Returns:
    - Interpolated RGB value as a numpy array of type np.uint8.
    """
    D12 = euclidean_distance(x1, y1, x2, y2)
    if D12 == 0:
        return np.array((np.array(I1) + np.array(I2)) // 2, dtype=np.uint8)
    
    D1 = euclidean_distance(x, y, x1, y1)
    D2 = euclidean_distance(x, y, x2, y2)
    
    if debug:
        print(f"Pixel di riferimento 1: ({x1}, {y1}), Valore: {I1}")
        print(f"Pixel di riferimento 2: ({x2}, {y2}), Valore: {I2}")
    
    # Calculate the difference between the reference pixels
    diff = np.linalg.norm(np.array(I1) - np.array(I2))
    
    # Determine how much weight to give to the surrounding mean based on the difference
    weight_factor = min(1, diff / max_diff_threshold)  # Ranges between 0 and 1
    interpolated_value = []
    
    for i in range(3):  # For each RGB channel
        In_channel = I2[i] * (D1 / D12) + I1[i] * (D2 / D12)

        # Blend the interpolated value with the surrounding mean based on weight_factor
        if surrounding_mean is not None:
            blended_value = surrounding_mean[i] * weight_factor + In_channel * (1 - weight_factor)
            In_channel_clipped = np.clip(blended_value, 0, 255)
        else:
            In_channel_clipped = np.clip(In_channel, 0, 255)
        
        interpolated_value.append(In_channel_clipped)
    
    return np.array(interpolated_value, dtype=np.uint8)




#%%

    
def surrounding_non_hair_mean(image, mask, x, y, radius=10) -> np.ndarray:
    """
    Calculate the mean RGB value of non-hair pixels surrounding a given pixel (x, y) within a radius.
    """
    rows, cols = mask.shape
    non_hair_pixels = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and mask[nx, ny] == 0:
                non_hair_pixels.append(image[nx, ny])
    
    if non_hair_pixels:
        return np.mean(non_hair_pixels, axis=0)
    else:
        # If no non-hair pixels are found, return a placeholder value
        return np.array([0, 0, 0], dtype=np.uint8)

#%%
def are_pixels_similar(I1, I2, threshold=50) -> bool:
    """
    Controlla se due pixel RGB (I1 e I2) sono simili entro una soglia data.
    La funzione restituisce True se la differenza assoluta per tutti i canali è inferiore alla soglia.
    """
    
    difference = np.abs(I1 - I2)
    #average between the two pixels
    average=(I1+I2)/2
    if(np.all(difference < threshold)):
        return True
    #return np.all(difference < threshold)
def check_pixel_similar_avg(x,y,I1,I2,surr,threshold=50):
    avg=I1+I2/2
    surrounding_difference = np.abs(avg - surr)
    if(np.all(surrounding_difference < threshold)):
        return True
    return False
#%%
  

def hair_region_interpolation(image: np.ndarray, mask: np.ndarray, max_length=50, min_length=10) -> np.ndarray:
    """
    Applies interpolation on all hair pixels in the mask to replace them with interpolated values.
    Optimized and adjusted to reduce white hair artifacts.
    """
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    max_rows, max_cols = mask.shape
    interpolated_image = image.copy()

    # Iterate over each pixel in the mask
    hair_pixels = np.column_stack(np.where(mask == 1))
    
    for (x, y) in hair_pixels:
        lengths = []

        # Calculate lengths of hair segments in each direction
        for dx, dy in directions:
            length = 0
            while (0 <= x + dx * length < max_rows and
                   0 <= y + dy * length < max_cols and
                   mask[x + dx * length, y + dy * length] == 1):
                length += 1
            lengths.append(length)

        max_line = max(lengths)
        other_lines = [l for l in lengths if l != max_line]

        # Check if the hair segment meets the length requirements
        if max_line >= max_length or not all(l < min_length for l in other_lines):
            continue

        # Find the index of the longest line
        max_index = lengths.index(max_line)

        # Calculate the perpendicular directions based on the longest line direction
        perpendicular_directions = [directions[(max_index + 2) % 8], directions[(max_index + 6) % 8]]
        surrounding = surrounding_non_hair_mean(image, mask, x, y)
        
        # Collect non-hair pixels in the perpendicular directions
        pixel_values_non_hair = []
        coordinates_non_hair = []
        intensity_threshold_bright = 190
        intensity_threshold_dark = 45
        for pdx, pdy in perpendicular_directions:
            nx, ny = x + pdx * 11, y + pdy * 11
            if (0 <= nx < max_rows) and (0 <= ny < max_cols) and mask[nx, ny] == 0:
                pixel_value = image[nx, ny]
                
                 # Calcola l'intensità del pixel
                if image.ndim == 2:  # Immagine in scala di grigi
                    intensity = pixel_value
                else:  # Immagine RGB
                    #intensity = np.mean(pixel_value)
                    intensity = 0.2989 * pixel_value[0] + 0.5870 * pixel_value[1] + 0.1140 * pixel_value[2]
                
                
                
                if intensity < intensity_threshold_bright and intensity>intensity_threshold_dark: #rla in modoand are_pixels_similar(I1,I2,threshold=40) and check_pixel_similar_avg(x,y,I1,I2,surrounding,threshold=40):
                    
                    pixel_values_non_hair.append(pixel_value)
                    coordinates_non_hair.append((nx, ny))

        # If two valid pixels are found, interpolate
        if len(pixel_values_non_hair) == 2:
            (x1, y1), (x2, y2) = coordinates_non_hair
            I1, I2 = pixel_values_non_hair

            # Perform bilinear interpolation for the current pixel
            In = bilinear_interpolation_rgb(x, y, x1, y1, I1, x2, y2,I2)
        else:
            # Fallback to the surrounding mean if no suitable pair is found
            In = surrounding.astype(np.uint8)

        # Apply the interpolated value to the pixel (x, y)
        interpolated_image[x, y] = In

    return interpolated_image





#%%
def enlarge_hair_mask(mask: np.ndarray, dilation_size=5) -> np.ndarray:
    """
    Enlarge the hair regions in the mask by applying dilation.
    """
    # Create a structuring element (kernel) for dilation, a square shape with given size
    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
    
    # Apply dilation to the mask to enlarge the hair regions
    enlarged_mask = cv2.dilate(mask, struct_element)
    
    return enlarged_mask

def binary_dilation(mask: np.ndarray, kernel_size=5):
    """
    Applica una dilatazione binaria alla maschera con un elemento strutturante 5x5.
    """
    # Crea un elemento strutturante di 5x5 con tutti i valori impostati a 1
    struct_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Applica la dilatazione binaria
    dilated_mask = cv2.dilate(mask, struct_element)
    return dilated_mask




def apply_median_filter(image: np.ndarray, mask: np.ndarray, dilation_size=5, kernel_size=5) -> np.ndarray:
    """
    Applies the median filter only to the enlarged hair regions identified by the mask.
    
    Parameters:
    - image: RGB image as a numpy array.
    - mask: Binary mask where hair regions are marked as 1.
    - dilation_size: Size of the dilation kernel to enlarge the hair regions.
    - kernel_size: Size of the kernel for the median filter (should be an odd number).
    
    Returns:
    - Filtered image with the median filter applied only to the enlarged hair regions.
    """
    # Enlarge the hair mask using dilation
    enlarged_mask = enlarge_hair_mask(mask, dilation_size)

    # Create a copy of the original image to preserve non-hair regions
    result_image = image.copy()

    # Apply median filter only on the enlarged hair regions by creating a masked version of the image
    for i in range(3):  # Apply per channel (R, G, B)
        # Extract the channel
        channel = image[:, :, i]

        # Apply median blur to the entire channel
        filtered_channel = cv2.medianBlur(channel, kernel_size)

        # Replace only the hair regions with the filtered values
        result_image[:, :, i][enlarged_mask == 1] = filtered_channel[enlarged_mask == 1]

    return result_image

#%%
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Helper function to ensure compatibility between image and mask
def ensure_compatibility(image: np.ndarray, mask: np.ndarray) ->bool:
    # Ensure the image is of type np.uint8 if not already
    if image.dtype != np.uint8:
        print(f"Converting image dtype from {image.dtype} to np.uint8")
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

    # Ensure the mask is binary (0 and 1) and of type np.uint8
    if mask.dtype != np.uint8 or np.any((mask != 0) & (mask != 1)):
        print("Normalizing mask to binary values (0 and 1) and converting to np.uint8")
        mask = (mask > 0).astype(np.uint8)
        #check the mask has only 0 and 1 values
        if np.any((mask != 0) & (mask != 1)):
            raise ValueError("Mask contains values other than 0 and 1")
    
    return True





#%%
path='../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000146.jpg'
im = Image.open(path)
        
plt.imshow(im)
plt.show()
#convert the image to an array
im_array = np.array(im)
#increase contrast
#im_array=rescale_intensity(im_array,5,95)


#%%

#apply the hair identification algorithm
mask=hair_identification_second_v(im_array, (8, 8))
#Show the mask
plt.imshow(mask)
plt.show()
#%%
ensure_compatibility(im_array,mask)
plt.imshow(im)
plt.show()


#%%
#im_interpolated=hair_region_interpolation(im_array,mask)
im_interpolated=try_function(im_array,mask)
#%%
#plot the image after interpolation and hair removal
plt.imshow(im_interpolated)
plt.show()
im_grey=cv2.cvtColor(im_interpolated, cv2.COLOR_RGB2GRAY)
#plot the grey scale image
plt.imshow(im_grey,cmap='gray')
plt.show()



#%%
#apply binary dilation to the mask and then the median filter to improve the result
dilated= enlarge_hair_mask(mask,5)
plt.imshow(dilated)
plt.show()

#%%
# Apply the dilated mask to interpolate the image
# Here, you need to re-run the interpolation algorithm using the dilated mask
im_dilated=hair_region_interpolation(im_interpolated,dilated)

# Plot the final image
plt.imshow(im_dilated)
plt.show()
#%%

#now i want to apply the median filter to the mask to smooth the regions of the hair
filtered_image=apply_median_filter(im_dilated,dilated,5)
plt.imshow(filtered_image)
plt.show()


#%%
def function_hair_pre_processing(path:str)->np.ndarray:
        im = Image.open(path)
        im_array = np.array(im)
        #plot the image
        plt.imshow(im)
        plt.show()
        #hair identification
        mask=hair_identification_second_v(im_array, (8, 8))
        plt.imshow(mask)
        plt.show()
        
        ensure_compatibility(im_array,mask)
        


        im_interpolated=hair_region_interpolation(im_array,mask)


        #plot the image after interpolation and hair removal
        plt.imshow(im_interpolated)
        plt.show()
        
        im_grey=cv2.cvtColor(im_interpolated, cv2.COLOR_RGB2GRAY)
        #plot the grey scale image
        plt.imshow(im_grey,cmap='gray')
        plt.show()




        #apply binary dilation to the mask and then the median filter to improve the result
        dilated= enlarge_hair_mask(mask,5)
        plt.imshow(dilated)
        plt.show()


        # Apply the dilated mask to interpolate the image
        #Here, you need to re-run the interpolation algorithm using the dilated mask
        im_dilated=hair_region_interpolation(im_interpolated,dilated)

        # Plot the final image
        plt.imshow(im_dilated)
        plt.show()


        #now i want to apply the median filter to the mask to smooth the regions of the hair
        filtered_image=apply_median_filter(im_dilated,dilated,5)
        plt.imshow(filtered_image)
        plt.show()


        return filtered_image

#%%
path='../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000146.jpg'
ima=function_hair_pre_processing(path)
plt.imshow(ima)
plt.show()



















    
    
    











# %%
