import numpy as np
from scipy.ndimage import median_filter
import cv2


class MedianFilter:
    '''This class applies a median filter to an image. The filter is a square mask of a given radius. We chose a radius of 3 as a compromise between the quality of the filtering and the computational time'''
    def __init__ (self, image: np.ndarray):
        self.im_array = image
        self.radius = 3
        self.filtered_image = self.median_filter_scipy()
    
    '''The following functions are used to apply the median filter to an image. 
    The first function uses the scipy library, the second one uses the cv2 library. 
    '''
    

    def median_filter_scipy(self)->np.ndarray:
        """
        This function can work in one dimension or in 3 if the image is RGB.
        We use it instead of separating channel by channel because it is faster.
        
        """
        if self.im_array.ndim==3:
            im_filtered = median_filter(self.im_array, size=(self.radius, self.radius, 1))
            return im_filtered
        else:
            return median_filter(self.im_array, size=(self.radius, self.radius))
        

    def median_filter_cv(self)->np.ndarray:
        """
        This function to apply the median filter is even faster than the previous one.
        """
        #convert to unit8
        im= (self.im_array*255).astype(np.uint8)    
        
        ksize = 2 * self.radius + 1  #We need a kernel size not pair
        return cv2.medianBlur(im, ksize)

    
    '''This were the original functions used, but they were less efficient than the previous ones to obtain the same result'''

    ''' def median_filter(self, im:np.ndarray, r=3, xy=None)->np.ndarray:
        """
        This function applies the median filter with a squared shape window with 
        dimension r*r.
        """
        lx = []
        ly = []
        
        if im.ndim != 2:
            raise ValueError("Median_filter works only with bidimensional images")
        
        (ty, tx) = im.shape  
        
        for k in range(-r, r + 1):
            for l in range(-r, r + 1):
                lx.append(k)
                ly.append(l)

        # compute the limits 
        debx = -min(lx)
        deby = -min(ly)
        finx = tx - max(lx)
        finy = ty - max(ly)
        ttx = finx - debx
        tty = finy - deby

        # matrix to store the values of the pixels
        tab = np.zeros((len(lx), ttx * tty))

        # fill the matrix with the values of the pixels
        for k in range(len(lx)):
            tab[k, :] = im[deby + ly[k]:deby + tty + ly[k], debx + lx[k]:debx + ttx + lx[k]].reshape(-1)

        # apply the median filter. out is the image with the filter applied so its type is the same as the input image
        out = im.copy()
        out[deby:finy, debx:finx] = np.median(tab, axis=0).reshape((tty, ttx))

        return out
        
        
        
        def check_median_filter(self, im:np.ndarray)->np.ndarray:
            #check the dimension of the image
            if( self.im_array.ndim == 3):
                #apply the median filter to each channel
                self.filtered_image = np.stack([median_filter(self.im_array[:,:,i], size=3) for i in range(3)], axis=2)
                return self.filtered_image
            else:
                self.filtered_image = self.median_filter(self.im_array, 3)
                return self.filtered_image        
            '''
        
    
        
            

           
        

    

    

