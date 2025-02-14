from scipy.ndimage import zoom
import os
import sys
import numpy as np
sys.path.append("../")
import matplotlib.image as mpimg
import manager
from manager import Manager
import matplotlib.pyplot as plt
import cv2 

if __name__=="__main__":
    folder_path =  "../dataset/skin_lesion_dataset-master/whole_ground/"
    alternative_path = "../dataset/skin_lesion_dataset-master/whole/"
    alt = sorted(os.listdir(folder_path))
    fol = sorted(os.listdir(alternative_path))

    arg_int = int(sys.argv[1])
    both = list(zip(alt, fol))[arg_int:arg_int+1]
    print(both)


    difficult_path = "../dataset/skin_lesion_dataset-master/difficult_ones2/"
    metrics_file = "../dataset/metrics.txt"
    total_metrics = 0
    totals = 0
    zoom_factor = (0.5, 0.5)
    manager = Manager(image_path = difficult_path, maximum_filter_size = 10, contrast_factor = 2, 
                    radius = 2, reduce_photo = True, zoom_factors = zoom_factor)
    for path in both:
        list_both = [alternative_path + path[1]] + [folder_path + path[0]]
        manager.change_path(list_both)
        output = manager.full_stack()
        met_list = []
        for key in output[1].keys():
            for metric in output[1][key].keys():
                if metric == "dice_score":
                    met_list.append(output[1][key][metric])
                    total_metrics += output[1][key][metric]
                    totals += 1
        file = open(metrics_file, "a")
        maximum = max(met_list)
        print(f"The maximum value is {maximum}")
        file.write(f"{key}: {maximum}\n")
        file.close()

