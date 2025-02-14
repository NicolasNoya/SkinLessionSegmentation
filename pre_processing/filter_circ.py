from scipy.ndimage import convolve
import numpy as np

class Circular_avg:
    '''This class applies a circular averaging filter to an image. 
    The filter is a circular mask of a given radius
    '''

    def __init__(self, image:np.ndarray, radius:int):
        self.im_array = image
        self.radius=radius
        self.filtered_image=self.apply_circular_avg_filter()
    

    def change_image(self, image:np.ndarray):
        """
        This function changes the image to which the filter is applied
        """
        self.im_array = image
        self.filtered_image=self.apply_circular_avg_filter()
 

    def circular_avg_filter(self)->np.ndarray:
        '''This function creates a circular filter of a given radius, creating a mask of 1 inside the circle and 0 outside'''
        h,w=np.ogrid[-self.radius:self.radius+1,-self.radius:self.radius+1] 
        mask = h**2+w**2<=self.radius**2
        mask_size=self.radius*2+1 
        #create an empty matrix to store the values
        kernel_val=np.zeros((mask_size,mask_size),dtype=float)
        #fill the matrix with 1 inside the circle
        kernel_val[mask]=1
        #I normalize so I won't change the intensity of the image. The sum of the values of the kernel is 1
        kernel_val=kernel_val/np.sum(kernel_val)
        return kernel_val


    def apply_circular_avg_filter(self)->np.ndarray:
        '''This fuction apply the filter: the filter will be applied to each pixel, to each channel'''
        im_array = np.array(self.im_array)
        filter=self.circular_avg_filter()
        #apply the filter using the convolve function
        filtered_image = np.stack([convolve(im_array[:,:,i], filter, mode='reflect') for i in range(3)], axis=-1)
        return filtered_image 


