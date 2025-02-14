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
        #I thought to apply the weights to the channels to give priority to the green channel, since it helps to identify the hair
        #MAYBE NOT NECESSARY
        weighted_difference = (weights[0] * difference_R + 
                           weights[1] * difference_G + 
                           weights[2] * difference_B)
        
        # Apply a threshold to obtain a binary mask. This threshold can be modified.
        _, binary_hair_mask = cv2.threshold(weighted_difference, 24, 1, cv2.THRESH_BINARY)

        # Convert to uint8 for visualization and further processing
        binary_hair_mask = binary_hair_mask.astype(np.uint8)

        return binary_hair_mask
    

    def euclidean_distance(self,x1, y1, x2, y2)->float:
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    

    def bilinear_interpolation(self,x, y, x1, y1, I1, x2, y2, I2)->float:
        D12=self.euclidean_distance(x1, y1, x2, y2)
        D1=self.euclidean_distance(x, y, x1, y1)
        D2=self.euclidean_distance(x, y, x2, y2)

        #Using the formula of the paper 
        In = I2 * (D1 / D12) + I1 * (D2 / D12) #intenisity in (x,y) pixel
        return In


    def bilinear_interpolation_rgb(self,x, y, x1, y1, I1, x2, y2, I2, surrounding_mean=None, max_diff_threshold=50, debug=False) -> np.ndarray:
        """
        Bilinear interpolation for an RGB pixel using two reference pixels.

        Parameters:
        - x, y: coordinates of the pixel to interpolate.
        - x1, y1, I1: coordinates and intensity of the first reference pixel.
        - x2, y2, I2: coordinates and intensity of the second reference pixel.
        - surrounding_mean: mean of surrounding non-hair pixels 
        - max_diff_threshold: maximum allowed difference between I1 and I2 before favoring the surrounding mean."""

        D12 = self.euclidean_distance(x1, y1, x2, y2)
        if D12 == 0:
            return np.array((np.array(I1) + np.array(I2)) // 2, dtype=np.uint8)
    
        D1 = self.euclidean_distance(x, y, x1, y1)
        D2 = self.euclidean_distance(x, y, x2, y2)
    
        # Calculate the difference between the reference pixels
        diff = np.linalg.norm(np.array(I1) - np.array(I2))
    
        # Determine how much weight to give to the surrounding mean based on the difference
        weight_factor = min(1, diff / max_diff_threshold)  # Ranges between 0 and 1
        interpolated_value = []
    
        for i in range(3):  # For each RGB channel
            In_channel = I2[i] * (D1 / D12) + I1[i] * (D2 / D12)

        #If the difference between the reference pixels is large, 
        # the algorithm chooses to rely more on the surrounding mean, 
        # assuming that the reference pixels are not reliable for interpolation.
        if surrounding_mean is not None and diff > max_diff_threshold:
            blended_value = surrounding_mean[i] * weight_factor + In_channel * (1 - weight_factor)
            In_channel_clipped = np.clip(blended_value, 50, 255)#I put 50 to avoid to have too dark pixels. THIS CAN ME MODIFIED 
        
        else:
            In_channel_clipped = np.clip(In_channel, 50, 255) #I put 50 to avoid to have too dark pixels
        
        interpolated_value.append(In_channel_clipped)
    
        return np.array(interpolated_value, dtype=np.uint8)

    ''' Function the that doesn't consider the offset 
    def surrounding_non_hair_mean2(self,image, mask, x, y, radius=3) -> np.ndarray:
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
        '''
        
    def surrounding_non_hair_mean(self, image, mask, x, y, offset=6, radius=3) -> np.ndarray:
        """This function calculates the mean RGB value of non-hair pixels surrounding a given pixel (x, y) within a radius, considering an offset to try to avoid the hair pixels.
        Parameters should be checked"""
        rows, cols = mask.shape
        non_hair_pixels = []

        # Define four potential offset positions far away from (x, y)
        offset_positions = [
            (x + offset, y),  # Right
            (x - offset, y),  # Left
            (x, y + offset),  # Down
            (x, y - offset)   # Up
        ]

        for ox, oy in offset_positions:
            # Make sure the offset positions are within image boundaries
            if 0 <= ox < rows and 0 <= oy < cols:
                # Now, search within a radius around this offset point
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx < rows and 0 <= ny < cols and mask[nx, ny] == 0:
                            non_hair_pixels.append(image[nx, ny])

        if non_hair_pixels:
            return np.mean(non_hair_pixels, axis=0)
        else:
        # If no non-hair pixels are found, return a placeholder value
            return np.array([0, 0, 0], dtype=np.uint8)


    def hair_region_interpolation(self,image: np.ndarray, mask: np.ndarray, max_length=50, min_length=10) -> np.ndarray:
        """
        Applies interpolation on all hair pixels in the mask (so only the supposed hairs) to replace them with interpolated values.
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

            #The paper suggests to consider only the case where the longest hair is longer than a certain threshold and all the other hairs are shorter than another threshold.

            if max_line> max_length and all(l < min_length for l in other_lines):
                continue

            # Find the index of the longest line
            max_index = lengths.index(max_line)

            # Calculate the perpendicular directions based on the longest line direction
            perpendicular_directions = [directions[(max_index + 2) % 8], directions[(max_index + 6) % 8]]
            surrounding = self.surrounding_non_hair_mean(image, mask, x, y)
        
            # Collect non-hair pixels in the perpendicular directions
            pixel_values_non_hair = []
            coordinates_non_hair = []
            
            #Selecting two pixels far away 11 pixels from the hair pixel, that are not hair pixels
            for pdx, pdy in perpendicular_directions:
                nx, ny = x + pdx * 11, y + pdy * 11
                if (0 <= nx < max_rows) and (0 <= ny < max_cols) and mask[nx, ny] == 0:
                    pixel_value = image[nx, ny]
                    pixel_values_non_hair.append(pixel_value)
                    coordinates_non_hair.append((nx, ny))
                
            # If two valid pixels are found, interpolate
            if len(pixel_values_non_hair) == 2:
                (x1, y1), (x2, y2) = coordinates_non_hair
                I1, I2 = pixel_values_non_hair
                   
                # Perform bilinear interpolation for the current pixel
                In = self.bilinear_interpolation_rgb(x, y, x1, y1, I1, x2, y2,I2)
            else:
                # Use to the surrounding mean if no suitable pair is found
                In = surrounding.astype(np.uint8)
            # Apply the interpolated value to the pixel (x, y)
            interpolated_image[x, y] = In
        
        return interpolated_image


    def enlarge_hair_mask(self,mask: np.ndarray, dilation_size=3) -> np.ndarray:

        """
        Enlarge the hair regions in the mask by applying dilation.
        """
        # Create a structuring element (kernel) for dilation, a square shape with given size
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        # Apply dilation to the mask
        enlarged_mask = cv2.dilate(mask, struct_element)
        return enlarged_mask


    def apply_median_filter(self,image: np.ndarray, mask: np.ndarray, dilation_size=3, kernel_size=3) -> np.ndarray:
        """
        Applies the median filter only to the enlarged hair regions identified by the mask.
    
        Parameters are:
        - image: RGB image as a numpy array.
        - mask: Binary mask where hair regions are marked as 1.
        - dilation_size: Size of the dilation kernel to enlarge the hair regions.
        - kernel_size: Size of the kernel for the median filter.
    
        """
        # Enlarge the hair mask using dilation
        enlarged_mask = self.enlarge_hair_mask(mask, dilation_size)

        # Create a copy of the original image to preserve non-hair regions
        result_image = image.copy()

        # Apply median filter to the entire image
        filtered_image = cv2.medianBlur(image, kernel_size)

        # Apply the mask to selectively replace only the hair regions
        result_image[enlarged_mask == 1] = filtered_image[enlarged_mask == 1]

        return result_image
    

    def function_hair_pre_processing(self)->np.ndarray:
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

