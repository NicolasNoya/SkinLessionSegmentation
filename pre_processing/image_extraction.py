import numpy as np
import torch


class ChannelExtractor:
    '''This class extracts the channels X, XoYoR, XoYoZoR, B and R from an image.
    The image, when necessary, is converted to the XoYoZ color space
    before extracting the channels'''
    def __init__(self, im_array:np.ndarray):
        self.t_matrix = torch.tensor([
            [0.4125, 0.3576, 0.1804],
            [0.2127, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9502]
        ], dtype=torch.float32)
        self.im_array = im_array
        self.converted_im=self.converter_rgb_to_XoYoZ()
        self.X_channel = self.extract_first_channel(self.converted_im)
        self.XoYoR_channel = self.extract_channel_XoYoR()
        self.XoYoZoR_channel = self.extract_channel_XoYoZoR()
        self.R_channel = self.extract_first_channel(self.im_array)
        self.B_channel =self.extract_channel_B(self.im_array)


    def extract_first_channel(self, img:np.ndarray)->np.ndarray:
        """
        This function extracts from, the given image, the first channel.
        The image passed as a parameter should be in RGB format.
        """
        first_channel = img[:,:,0]
        return first_channel

    
    def converter_rgb_to_XoYoZ(self)->np.ndarray:
        """
        This function converts the given image from RGB to XoYoZ color space.
        """
        r,c,d=self.im_array.shape
        im_conv = np.zeros((r * c, self.t_matrix.shape[0]))#initialize the image with same shape 
        #I want to reduce the image to two dimensions
        im = self.im_array.reshape(r*c,d)
        #Multiply by the transformation matrix using numpy function that apply the matrix to all pixels
        im_conv=np.dot(im,self.t_matrix.T) 
        
        #multiply by the transformation matrix using a for cycle to apply the matrix to all pixels was not efficient

        #reshape into the original shape
        im_conv = im_conv.reshape(r,c,self.t_matrix.shape[0])
        return im_conv 
        
        
    def extract_channel_XoYoR(self)->np.ndarray:
        """
        This function computes the XoYoR channel from the given image.
        """
        XoYoR_channel = self.converted_im[:,:,1]+ self.converted_im[:,:,2]+self.im_array[:,:,0]
        return XoYoR_channel
    
        
    def extract_channel_XoYoZoR(self)->np.ndarray:
        """
        This function computes the XoYoZoR channel from the given image.
        """
        XoYoZoR_channel = self.converted_im[:,:,0]+self.converted_im[:,:,1]+self.converted_im[:,:,2]+self.im_array[:,:,0]
        return XoYoZoR_channel 


    def extract_channel_B(self, im:np.ndarray)->np.ndarray:
        """
        This function computes the B channel from the given image.
        """
        B_channel = im[:,:,2]
        return B_channel
        
        

    
        
        







    
    
    
        
    
        
            

           
        

    

    

