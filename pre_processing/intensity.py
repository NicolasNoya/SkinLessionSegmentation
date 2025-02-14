import numpy as np


class IntensityAdjustment :
    '''This class is made to rescale an image using percentiles, 
    in order to improve the contrast before applying thresholding
    '''
    def __init__ (self, low_percentile: int =1, high_percentile: int=99):

        # This values are fixed by the paper, then they not hiperparameters
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile


    def rescale_intensity(self, image: np.ndarray)->np.ndarray:
        '''This function is going to rescale the image intensity to the range 0-1, 
        so that the low percentile will be 0 and the high percentile will be 1
        '''
        low_value=np.percentile(image, self.low_percentile)
        high_value=np.percentile(image, self.high_percentile)
        #rescale the image to the range 0-1,mapping each value to the new range
        im_rescaled=(image-low_value)/(high_value-low_value)
        return im_rescaled #the return value is an array of the rescaled image




