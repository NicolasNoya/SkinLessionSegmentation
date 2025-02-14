import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""This class is used to remove hair from images using morphological operations and
interpolation."""
class HairRemoval:
    def __init__ (self, im_path:str):
        self.img = Image.open(im_path)
        self.im_array = np.array(self.img)


    def change_img(self,im_path:str):
        self.img = Image.open(im_path)
        self.im_array = np.array(self.img)


    def grayscale_morphological_close(self,im:np.ndarray, kernel:np.ndarray)->np.ndarray:
        '''This function applies two steps: first the dilation and then the erosion to a grayscale image'''
        
        dilated_im=cv2.dilate(im, kernel, iterations=1)
        eroded_im=cv2.erode(dilated_im, kernel, iterations=1)
        return eroded_im

    
    def create_rotated_ellipse(self,kernel_size, angle)->np.ndarray:
        """Creates a rotated elliptical structural element."""
        #Empty image 
        ellipse_img = np.zeros((kernel_size[0], kernel_size[1]), dtype=np.uint8)
    
        #ellipse construction centered in the middle of the image
        center = (kernel_size[0] // 2, kernel_size[1] // 2)
        axes = (kernel_size[0] // 2, kernel_size[1] // 2)
        cv2.ellipse(ellipse_img, center, axes, angle, 0, 360, 1, -1)

        return ellipse_img
    

    def hair_identification_second_v (self,im: np.ndarray,kernel_size=(7,7), weights=(0.2, 0.7, 0.1)) -> np.ndarray:
        '''Apply the morphological closing to RGB image. The idea was to use different structure elements for the directions'''
    
        # Creating structural elements
        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
        

        diagonal_ellipse= self.create_rotated_ellipse(kernel_size, 45)
        diagonal_ellipse_1 = self.create_rotated_ellipse(kernel_size, 30)
        diagonal_ellipse_2 = self.create_rotated_ellipse(kernel_size, 60)
        diagonal_ellipse_3 = self.create_rotated_ellipse(kernel_size, 90)
        large_diagonal_ellipse = self.create_rotated_ellipse((kernel_size[0] * 2, kernel_size[1] * 2), 45)

        # Applying morphological closing for each direction
        def apply_closing_for_all_directions(channel): 
            horizontal_close = self.grayscale_morphological_close(channel, horizontal)
            vertical_close = self.grayscale_morphological_close(channel, vertical)
    
            diagonal_close = self.grayscale_morphological_close(channel, diagonal_ellipse)
            diago1=self.grayscale_morphological_close(channel, diagonal_ellipse_1)
            diago2=self.grayscale_morphological_close(channel, diagonal_ellipse_2)
            diago3=self.grayscale_morphological_close(channel, diagonal_ellipse_3)
            diago_large=self.grayscale_morphological_close(channel, large_diagonal_ellipse)
            return np.maximum.reduce([horizontal_close, vertical_close, diagonal_close,diago1,diago2,diago3,diago_large]) 

        # Apply morphological closing to all RGB channels
        BGR_max = np.dstack([apply_closing_for_all_directions(im[:, :, i]) for i in range(3)])

        # Compute the absolute difference between the original image and the morphologically closed image
        difference = cv2.absdiff(im, BGR_max)

        #Extracting the difference between the original image and the morphologically closed image for each channel
        difference_B, difference_G, difference_R = difference[:, :, 0], difference[:, :, 1], difference[:, :, 2]
        #Apply the weights to the channels to give priority to the green channel, since it helps to identify the hair
        weighted_difference = (weights[0] * difference_R + 
                           weights[1] * difference_G + 
                           weights[2] * difference_B)
        
        # Apply a threshold to obtain a binary mask. This threshold can be modified.
        _, binary_hair_mask = cv2.threshold(weighted_difference, 24, 1, cv2.THRESH_BINARY)

        # Convert to uint8 for visualization and further processing
        binary_hair_mask = binary_hair_mask.astype(np.uint8)

        return binary_hair_mask
    

    def enlarge_hair_mask(self,mask: np.ndarray, dilation_size=3) -> np.ndarray:

        """
        Enlarge the hair regions in the mask by applying dilation.
        """
        # Create a structuring element (kernel) for dilation, a square shape with given size
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        # Apply dilation to the mask
        enlarged_mask = cv2.dilate(mask, struct_element)
        return enlarged_mask


    def function_hair_pre_processing(self)->np.ndarray:
        """
        This function applies the whole stack of hair removal process to the image in
        self.img.
        """
        im = self.img
        im_array = self.im_array
    
        #hair identification
        mask=self.hair_identification_second_v(im_array, (7, 7))
        # im_interpolated=self.hair_region_interpolation(im_array,mask)
        
        #apply binary dilation to the mask and then the median filter to improve the result
        dilated_mask= self.enlarge_hair_mask(mask,3)

        dilated_mask_3 = []
        dilated_mask_3.append(dilated_mask)
        dilated_mask_3.append(dilated_mask)
        dilated_mask_3.append(dilated_mask)
        dilated_mask_3 = np.array(dilated_mask_3).transpose(1,2,0)
        final_image = np.where(dilated_mask_3 == 1, 255, im_array)
        
        return final_image/255

