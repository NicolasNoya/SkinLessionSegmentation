from typing import Tuple, Dict, Union, List
from enum import Enum
import cv2
import numpy as np
import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

sys.path.append('./segmentation')
sys.path.append('./metrics')
sys.path.append('./pre_processing')

from segmentation.segmentation_class import Segmentator
from metrics.metrics import Metrics
from pre_processing.filter_circ import Circular_avg
from pre_processing.image_extraction import ChannelExtractor
from pre_processing.intensity import IntensityAdjustment
from pre_processing.hair_removal import HairRemoval
from scipy.ndimage import zoom
from pre_processing.median_filter import MedianFilter


class Channel(Enum):
    "X_channel"
    "XoYoR_channel"
    "XoYoZoR_channel"
    "R_channel"
    "B_channel"

class Metric(Enum):
    "dice_score"
    "jaccard_score"
    "recall"
    "precision"
    "f1"
    "total"

class Manager:
    def __init__(self, image_path: Union[str, List[str]]=None, **kwargs):
        """
        This function loads a set of images from a given path and returns
        the predicted masks.
        In the kwargs, the user can specify the hiperparameters of the model.
        The kwargs parameters are:
            - filter_coefficient,
            - radius,
            - reduce_photo,
            - zoom_factor,
            - low_percentile,
            - high_percentile,
            - last_max
            if not specified the default values will be used.
        """
        self.image_path = image_path
        self.segmentator = Segmentator()
        self.metrics = Metrics(np.zeros((100,100)), np.zeros((100,100)))
        self.full_stack_executed = False
        if image_path:
            if isinstance(image_path, str):
                self.hair_removal = HairRemoval(image_path + os.listdir(image_path)[0])
            else:
                self.hair_removal = HairRemoval(image_path[0])

        if  kwargs:
            self.filter_coefficient = (kwargs['filter_coefficient'] if 'filter_coefficient' in kwargs else 15)
            self.radius = (kwargs['radius'] if 'radius' in kwargs else 5)
            self.reduce_photo = (kwargs['reduce_photo'] if 'reduce_photo' in kwargs else False)
            self.zoom_factors = (kwargs['zoom_factors'] if 'zoom_factors' in kwargs else (0.5, 0.5))
            self.low_percentile = (kwargs['low_percentile'] if 'low_percentile' in kwargs else 1)
            self.high_percentile = (kwargs['high_percentile'] if 'high_percentile' in kwargs else 99)
            self.last_max = (kwargs['last_max'] if 'last_max' in kwargs else 4)
        else:
            self.filter_coefficient = 15
            self.radius = 2
            self.reduce_photo = False
            self.zoom_factors = (1, 1)
            self.low_percentile = 1
            self.high_percentile = 99
            self.last_max = 4


    def change_path(self, image_path: Union[str, List[str]]):
        """
        This function changes the path of the images to be loaded.
        """
        self.image_path = image_path


    def full_stack(self)->Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
        """
        This function loads a set of images and computes the whole stack of the images. 
        Returns a dictionary with the name of the image and their predicted masks 
        and a dictionary with the metrics.
        
        Returns:
            - masks_dict: Dict[str, Dict[str, np.ndarray]]
            - metrics_dict: Dict[str, Dict[str, float]]
        Example:
            masks_dict = {00030: {"Original": np.ndarray, "predicted": {"X_channel": np.ndarray,...}}}
            metrics_dict = {"X_channel": {"dice_score": 0.5, "jaccard_score": 0.5,...}}
        """
        self.full_stack_executed = True
        self.masks_dict = {}
        if isinstance(self.image_path, str):
            dirlist = os.listdir(self.image_path)
            dirlist = [f'{self.image_path}/{i}' for i in dirlist]
        else:
            dirlist = self.image_path

        for path in dirlist:
            dict_key = path.split("/")[-1].split("_")[1].split(".")[0]
            if dict_key not in list(self.masks_dict.keys()):
                self.masks_dict[dict_key] = {}

            if "Segmentation" in path:
                img = mpimg.imread(path).transpose()
                if self.reduce_photo:
                    img = zoom(img, self.zoom_factors)
                self.masks_dict[dict_key]["Original"]=img.astype(np.uint8)
            else:
                self.hair_removal.change_img(path)
                img = self.hair_removal.function_hair_pre_processing()
                if self.reduce_photo:
                    zoom_factors = (self.zoom_factors[0], self.zoom_factors[1], 1)
                    img = zoom(img, zoom_factors)

                self.masks_dict[dict_key]["predicted"]=self.segmentation(img)

        self.metrics_dict = self.test_metrics(self.masks_dict) 

        return (self.masks_dict, self.metrics_dict)

    
    def plot_metric(self, metric: Metric, channel: Channel = None):
        """
        This function prints the metrics in the terminal.
        """
        if channel is None:
            for key in self.metrics_dict.keys():
                print(f"{key}: {self.metrics_dict[key][metric]}")
        else: 
            print(f"{channel}: {self.metrics_dict[channel][metric]}")
    

    def preprocess(self, image: np.ndarray):
        """
        This function takes an image and returns a dictionary with the preprocessed image as value
        and the name of the preprocessed transformation applied as key.
        The preprocessed transformations are:
            - X_channel
            - XoYoR_channel
            - XoYoZoR_channel
            - R_channel
            - B_channel
        """
        filtered_image = image

        #apply the median filter
        filter_median=MedianFilter(image)
        filtered_image=filter_median.filtered_image
        circular_filt_image = filtered_image

        #apply the circular averaging filter
        filter_circular=Circular_avg(filtered_image, self.radius)
        circular_filt_image=filter_circular.filtered_image

        #apply the channel extractor and obtain the 4 possible channels
        channel_extractor=ChannelExtractor(circular_filt_image)

        X_channel = channel_extractor.X_channel
        XoYoR_channel = channel_extractor.XoYoR_channel
        XoYoZoR_channel = channel_extractor.XoYoZoR_channel
        R_channel = channel_extractor.R_channel
        B_channel = channel_extractor.B_channel

        # adjust the intensity of the image
        intensity_adjustment=IntensityAdjustment(self.low_percentile, self.high_percentile)
        X_adjusted_im=intensity_adjustment.rescale_intensity(X_channel)
        XoYoR_adjusted_im=intensity_adjustment.rescale_intensity(XoYoR_channel)
        XoYoZoR_adjusted_im=intensity_adjustment.rescale_intensity(XoYoZoR_channel)
        R_adjusted_im=intensity_adjustment.rescale_intensity(R_channel)
        B_adjusted_im=intensity_adjustment.rescale_intensity(B_channel)

        self.X_channel = X_adjusted_im
        self.XoYoR_channel = XoYoR_adjusted_im
        self.XoYoZoR_channel = XoYoZoR_adjusted_im
        self.R_channel = R_adjusted_im
        self.B_channel = B_adjusted_im

        return {
            "X_channel": X_adjusted_im,
            "XoYoR_channel": XoYoR_adjusted_im,
            "XoYoZoR_channel": XoYoZoR_adjusted_im,
            "R_channel": R_adjusted_im,
            "B_channel": B_adjusted_im
        }


    def segmentation(self, image: np.ndarray)->Dict[str, np.ndarray]:
        """
        This function takes an image, preprocesses it and returns the predicted mask of 
        each channel in a dictionary. 
        The names of the channels are the same that thouse in self.preprocess.
        """
        preprocessed_images = self.preprocess(image)
        
        segmentation_dictionary = {}

        for key, ima in preprocessed_images.items():
            segmented_image =  self.segmentator.whole_stack(ima, self.filter_coefficient, last_max=self.last_max)
            segmentation_dictionary[key] = segmented_image
        return segmentation_dictionary


    def test_metrics(self, masks_dict: dict)->Dict[str, Dict[str, float]]:
        """
        This function takes a dictionary with the original masks and the predicted masks 
        for each channel and returns a dictionary with the average of every metrics for each channel.
        """
        final_dict = {}
        for key in list(masks_dict[list(masks_dict.keys())[0]]["predicted"].keys()):
            final_dict[key] = {
            "dice_score" : 0.0,
            "jaccard_score" : 0.0,
            "recall" : 0.0,
            "precision" : 0.0,
            "f1" : 0.0,
            "total" : 0.0,
            }

        metrics = Metrics(np.zeros((1,1)), np.zeros((1,1)))
        for key in self.masks_dict.keys():
            y_true = self.masks_dict[key]["Original"]
            for channel, y_pred in self.masks_dict[key]["predicted"].items():
                if y_true.shape != y_pred.shape:
                    y_true = y_true.transpose()
                metrics.redefine_sample(y_true, y_pred)
                final_dict[channel]["dice_score"] += metrics.dice_score()
                final_dict[channel]["jaccard_score"] += metrics.jaccard_score()
                final_dict[channel]["recall"] += metrics.recall()
                final_dict[channel]["precision"] += metrics.precision()
                final_dict[channel]["f1"] += metrics.f1_score()
                final_dict[channel]["total"] += 1
        
        for key in final_dict.keys():
            for metric in final_dict[key].keys():
                if metric != "total":
                    final_dict[key][metric] = final_dict[key][metric]/final_dict[key]["total"]
        
        return final_dict
    
    def create_borders(self, channel: Channel):
        """
        This function creates two borders, black for the predicted mask and 
        red for the original one and add them to the original image. It also
        plots the image for it to be analyzed. 

        Note: This function is only for debugging.
        SHOULD BE EXECUTED ONLY AFTER full_stack
        """
        class ExecuteError(Exception):
            pass

        if not self.full_stack_executed:
            raise ExecuteError("Execute full_stack before this procedure.")

        if isinstance(self.image_path, list):
            path = "/".join(self.image_path[0].split("/")[:-1])
        else:
            path =  self.image_path


        for key in self.masks_dict.keys():
            original_image = mpimg.imread(f"{path}/ISIC_{key}.jpg").transpose()/255
            if self.reduce_photo:
                original_image = zoom(original_image, (1, self.zoom_factors[0],self.zoom_factors[1]))
            original_mask = self.masks_dict[key]["Original"].transpose()
            predicted_mask = self.masks_dict[key]["predicted"][channel]
            predicted_edges = cv2.Canny(predicted_mask, 0, 1)
            original_edges = cv2.Canny(original_mask, 0, 1)
            transposed = []
            transposed.append(original_image[0] - predicted_edges.transpose() + original_edges.transpose())
            transposed.append(original_image[1] - predicted_edges.transpose() - original_edges.transpose())
            transposed.append(original_image[2] - predicted_edges.transpose() - original_edges.transpose())
            plt.imshow(np.array(transposed).transpose() )
            plt.show()
    
    def predict(self, path: str):
        """
        This function takes an image in a path, apply the whole preprocessing, segmentation and postprocessing
        and returns a dictionary with the predicted mask for every channel.
        """
        self.hair_removal.change_img(path)
        img = self.hair_removal.function_hair_pre_processing()
        return self.segmentation(img)


