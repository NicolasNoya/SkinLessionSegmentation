#%%
import sys
sys.path.append('../segmentation')
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from segmentation.segmentation import Segmentator
from metrics.metrics import Metrics
import os

#%%

masks_dict = {}
segmentator = Segmentator()
melanoma = "prueba"
for i in os.listdir(f'./dataset/skin_lesion_dataset-master/{melanoma}'):
    path = f'./dataset/skin_lesion_dataset-master/{melanoma}/{i}'
    dict_key = i.split("_")[1].split(".")[0]
    if dict_key not in list(masks_dict.keys()):
        masks_dict[dict_key] = {}

    if "Segmentation" in i:
        img = mpimg.imread(path).transpose()
        masks_dict[dict_key]["Original"]=img.astype(np.uint8)
        print(f"Original image for {dict_key}")
        # plt.imshow(masks_dict[dict_key]["Original"], cmap='gray')
        # plt.show()
    else:
        img = mpimg.imread(path)/255
        masks_dict[dict_key]["predicted"]=segmentator.whole_stack(img, 2, 30)
        print(f"Predicted image for {dict_key}")

#%%
for key in masks_dict.keys():
    original_image = masks_dict[key]["Original"]
    print(original_image.shape)

    final_image = masks_dict[key]["predicted"]
    print(final_image.shape)

#%%
total_metrics = {
"total_dice" : 0,
"total_jaccard" : 0,
"total_recall" : 0,
"total_precision" : 0,
"total_f1" : 0,
"total" : 0
}
metrics = Metrics(np.zeros((100,100)), np.zeros((100,100)))
for key in masks_dict.keys():
    metrics.redefine_sample(masks_dict[key]["Original"], masks_dict[key]["predicted"])
    y_true = masks_dict[key]["Original"]
    y_pred = masks_dict[key]["predicted"]
    plt.imshow(y_true.transpose(), cmap='gray')
    plt.show()
    plt.imshow(y_pred.transpose(), cmap='gray')
    plt.show()
    total_metrics["total_dice"] += metrics.dice_score()
    total_metrics["total_jaccard"] += metrics.jaccard_score()
    total_metrics["total_recall"] += metrics.recall()
    total_metrics["total_precision"] += metrics.precision()
    total_metrics["total_f1"] += metrics.f1_score()
    total_metrics["total"] += 1

#%%
for met in total_metrics.keys():
    if met != "total":
        print(f"{met}: {total_metrics[met]/total_metrics['total']}")

# %%
print(total_metrics["total"])
