#%%
from scipy.ndimage import convolve
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from scipy.ndimage import median_filter
import cv2
#%%
'''Importing the classes and functions from the files '''
from image_extraction import ChannelExtractor

from filter_circ import Circular_avg

from intensity import IntensityAdjustment

from median_filter import MedianFilter

#%%
'''The image is given always as a numpy array with values between 0 and 1'''
#use a for cycle to read all the images in the dataset and pre-process all of them 
for i in os.listdir('../dataset/skin_lesion_dataset-master/melanoma'):
    
    if i.endswith('.jpg'):
        img = mpimg.imread(f'../dataset/skin_lesion_dataset-master/melanoma/{i}')  

        #normalize the image before manipulating it
        #convert the image to a numpy array
        img = np.array(img)
        img = img/255
        plt.imshow(img)
        plt.show()
        img_path=f'../dataset/skin_lesion_dataset-master/melanoma/{i}'
        #istantiate the classes for each image
        channel_extractor=ChannelExtractor(img_path)
        filter_circular=Circular_avg(img_path)
        intensity_adjustment=IntensityAdjustment(img_path)
        filter_median=MedianFilter(img_path)

        #apply the median filter
        filtered_image=filter_median.median_filter_scipy(img, 1, 3)
        #filtered_image=filter_median.median_filter_scipy(img, 1, 3)
        plt.imshow(filtered_image)
        plt.show()

        #apply the circular averaging filter
        circular_filt_image=filter_circular.apply_circular_avg_filter(5,filtered_image)
        #circular_filt_image=filter_circular.apply_circular_avg_filter(5,filtered_image)
        plt.imshow(circular_filt_image)
        plt.show()

        #apply the channel extractor and obtain the 4 possible channels

        converted_im=channel_extractor.converter_rgb_to_XoYoZ(img)
        X_channel = channel_extractor.extract_first_channel(converted_im)
        XoYoR_channel = channel_extractor.extract_channel_XoYoR(img)
        XoYoZoR_channel = channel_extractor.extract_channel_XoYoZoR(img)
        R_channel = channel_extractor.extract_first_channel(img)
        B_channel=channel_extractor.extract_channel_B(img)


        #show them on grey scale
        plt.imshow(X_channel, cmap='gray')
        plt.show()
        plt.imshow(XoYoR_channel, cmap='gray')
        plt.show()
        plt.imshow(XoYoZoR_channel, cmap='gray')
        plt.show()
        plt.imshow(R_channel, cmap='gray')
        plt.show()
        plt.imshow(B_channel, cmap='gray')
        plt.show()


        '''Applying intenisity adjustment to the image'''
        X_adjusted_im=intensity_adjustment.rescale_intensity(X_channel,1,99)
        XoYoR_adjusted_im=intensity_adjustment.rescale_intensity(XoYoR_channel,1,99)
        XoYoZoR_adjusted_im=intensity_adjustment.rescale_intensity(XoYoZoR_channel,1,99)
        R_adjusted_im=intensity_adjustment.rescale_intensity(R_channel,1,99)
        B_adjusted_im=intensity_adjustment.rescale_intensity(B_channel,1,99)

        #show them on grey scale
        plt.imshow(X_adjusted_im, cmap='gray')
        plt.show()
        plt.imshow(XoYoR_adjusted_im, cmap='gray')
        plt.show()
        plt.imshow(XoYoZoR_adjusted_im, cmap='gray')
        plt.show()
        plt.imshow(R_adjusted_im, cmap='gray')
        plt.show()
        plt.imshow(B_adjusted_im, cmap='gray')
        plt.show()

        #I want to compare the rescaled image with the X_channel
        plt.imshow(X_channel, cmap='gray')
        plt.show()
        plt.imshow(X_adjusted_im, cmap='gray')
        plt.show()
        #compute a difference image to check if the rescaling was done correctly
        #diff_im=X_channel-X_adjusted_im
        #plt.imshow(diff_im, cmap='gray')
        #plt.show()




# %%
def applying_pre_processing(img_path:str)->list:
    '''This function takes an image path and returns the pre-processed image'''
    img=mpimg.imread(img_path)
    img = np.array(img)
    img = img/255
    plt.imshow(img)
    plt.show()

    #istantiate the classes for each image
    channel_extractor=ChannelExtractor(img_path)
    filter_circular=Circular_avg(img_path)
    intensity_adjustment=IntensityAdjustment(img_path)
    filter_median=MedianFilter(img_path)

    #apply the median filter
    filtered_image=filter_median.median_filter_scipy(img, 1, 3)
    #filtered_image=filter_median.median_filter_scipy(img, 1, 3)
    plt.imshow(filtered_image)
    plt.show()

    #apply the circular averaging filter
    circular_filt_image=filter_circular.apply_circular_avg_filter(5,filtered_image)
    #circular_filt_image=filter_circular.apply_circular_avg_filter(5,filtered_image)
    plt.imshow(circular_filt_image)
    plt.show()

    #apply the channel extractor and obtain the 4 possible channels

    converted_im=channel_extractor.converter_rgb_to_XoYoZ(img)
    X_channel = channel_extractor.extract_first_channel(converted_im)
    XoYoR_channel = channel_extractor.extract_channel_XoYoR(img)
    XoYoZoR_channel = channel_extractor.extract_channel_XoYoZoR(img)
    R_channel = channel_extractor.extract_first_channel(img)
    B_channel=channel_extractor.extract_channel_B(img)


    #show them on grey scale
    plt.imshow(X_channel, cmap='gray')
    plt.show()
    plt.imshow(XoYoR_channel, cmap='gray')
    plt.show()
    plt.imshow(XoYoZoR_channel, cmap='gray')
    plt.show()
    plt.imshow(R_channel, cmap='gray')
    plt.show()
    plt.imshow(B_channel, cmap='gray')
    plt.show()

    '''Applying intenisity adjustment to the image'''
    X_adjusted_im=intensity_adjustment.rescale_intensity(X_channel,1,99)
    XoYoR_adjusted_im=intensity_adjustment.rescale_intensity(XoYoR_channel,1,99)
    XoYoZoR_adjusted_im=intensity_adjustment.rescale_intensity(XoYoZoR_channel,1,99)
    R_adjusted_im=intensity_adjustment.rescale_intensity(R_channel,1,99)
    B_adjusted_im=intensity_adjustment.rescale_intensity(B_channel,1,99)

    #show them on grey scale
    plt.imshow(X_adjusted_im, cmap='gray')
    plt.show()
    plt.imshow(XoYoR_adjusted_im, cmap='gray')
    plt.show()
    plt.imshow(XoYoZoR_adjusted_im, cmap='gray')
    plt.show()
    plt.imshow(R_adjusted_im, cmap='gray')
    plt.show()
    plt.imshow(B_adjusted_im, cmap='gray')
    plt.show()

    return [X_adjusted_im, XoYoR_adjusted_im, XoYoZoR_adjusted_im, R_adjusted_im,B_adjusted_im]

    # %%
#trying to apply the pre-processing function to an image 
img_path='../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000046.jpg'
pre_processed_images=applying_pre_processing(img_path)

    
# %%
for img in pre_processed_images:
    plt.imshow(img, cmap='gray')
    plt.show()

# %%
