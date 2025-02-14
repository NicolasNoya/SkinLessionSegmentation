from typing import Tuple
import math
from skimage import measure
import skimage.morphology as morpho
import numpy as np
from scipy.ndimage import binary_fill_holes, maximum_filter

class Segmentator:
    def __init__(self):
        pass


    def apply_a_mask(self, image: np.ndarray, mask: np.ndarray)->np.ndarray:
        """
        This function applies a mask to an image and returns the masked image. 
        This function expects tridimensional image and a bidimensional mask.
        If the image is bidimensional a dimension of value 1 will be added in the first place.
        (256, 432) -> (1, 256, 432) 
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) > 3:
            raise ValueError('Images must have 3 dimensions, image batches are not supported')

        if image.shape[0] > 4 and len(image.shape) > 2:
            image = image.transpose()
 
        for i in range(image.shape[0]):
            image[i] = image[i] * mask

        return image

    def otsus_function(self, image: np.ndarray)->np.ndarray:
        """
        This functions returns the thresholded mask obtained by the Otsu's method. 

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
            grey_hist = np.histogram(channel, bins=256, range=(0, 1))
            uzip = list(zip(grey_hist[0], (grey_hist[1]).astype(np.float32)))
            new_sigma = -1
            for t in range(1, 254):
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
                    new_sigma = sigma_b
                    t_max = t
            
            thresholds.append(grey_hist[1][t_max])
        
        for index, threshold in enumerate(thresholds):
            masks.append(self.apply_a_threshold(image[index], threshold))
                    
        sum_mask = masks[0]
    
        for channel in masks[0:]:
            sum_mask += channel

        final_mask = np.where(sum_mask>0, 1, 0)

        return final_mask

    
    def apply_a_threshold(self, image:np.ndarray, threshold:int)->np.ndarray:
        """
        This function applies a threshold to the image and returns a mask with 
        the pixels above the threshold in white and the rest in black.
        
        Input:
        - image: The image to be thresholded, this image should 
            have only one channel.
        Output:
        - mask: The mask with the pixels above the threshold in white and the rest
            in black.
        """
        mask = np.where(image>threshold, 0, 1)
        final_mask = mask * image
        return final_mask


    def circular_mask(self, img_size: Tuple[int, int], radious: int):
        """
        Creates a circular mask with values of 1 inside a radious 'r' from the center.
        This mask will be used to filter the cornes in the images.
        """
        h, w = img_size
        Y, X = np.ogrid[:h, :w]

        centre_y, centre_x = h // 2, w // 2
        centre_distance = (X - centre_x)**2 + (Y - centre_y)**2
        mask = centre_distance < (radious**2)
        
        return mask.astype(int)  # Convertir a entero para tener 0 y 1


    def preprocess_corners(self, image: np.ndarray, radious: int = 80)->np.ndarray:
        """
        This function preprocess the corners of the image and returns a mask with 
        the corners with value 0 and the rest with value 1. This will be then multiplied 
        elemnt by element of the matrix to apply the mask.
        """
        if image.shape[0] > 4 and len(image.shape) > 2:
            image = image.transpose()
            dimension = (image.shape[1], image.shape[2])
        else:
            dimension = (image.shape[0], image.shape[1])

        circular_mask = self.circular_mask(dimension, radious=radious)

        if image.shape[0] > 5 and len(image.shape) > 2:
            for i in range(image.shape[0]):
                image[i] = image[i] * circular_mask
        else:
            image = image * circular_mask

        return image
    

    def increase_contrast(self, image: np.ndarray, contrast_factor: float = 1.5)->np.ndarray:
        """
        This function increases the contrast of the image.
        """
        if image.shape[0] > 5 and len(image.shape) > 2:
            image = image.transpose()

        mean = image.mean()

        image_contrasted = contrast_factor * (image - mean) + mean
        return image_contrasted

    
    def separate_individual_objects(self, mask: np.ndarray)->np.ndarray:
        """
        This function is used to separate individula objects from the mask and 
        also returns the mask with the biggest object (that is presumibly the lesion).
        """
        # Label the connected components
        labeled_mask, num_labels = measure.label(mask, background=0, return_num=True)

        # Generate individual masks for each object
        individual_masks = [(labeled_mask == i).astype(np.uint8) for i in range(1, num_labels + 1)]

        # Get the maximum mask
        masks_size = [mask.sum() for mask in individual_masks]
        max_mask = masks_size.index(max(masks_size))

        return individual_masks[max_mask]


    def whole_stack(self, img: np.ndarray, filter_coefficient: int = 15, last_max: int = 4)->np.ndarray:
        """
        This function applies the whole stack of 
        functions to an image and returns the final image with just the lession.
        """
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
 
        transpose_img = img.transpose()
        thresholded_image = self.otsus_function(transpose_img)
        final_image = self.preprocess_corners(thresholded_image, (thresholded_image.shape[1]-200)/2) 
        final_image = binary_fill_holes(final_image)
        size = int(math.sqrt(np.sum(final_image))/filter_coefficient)
        final_image = maximum_filter(final_image , size=int(size/filter_coefficient))
        final_image = self.separate_individual_objects(final_image)
        final_image = maximum_filter(final_image , size=int(size))
        final_image = morpho.closing(final_image, morpho.rectangle(last_max, last_max))
        return final_image

   
    