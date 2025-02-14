#%%
#List of scores:
# 72.44
# 72.44
# 73 con closing
from scipy.ndimage import zoom
import os
import sys
import numpy as np
sys.path.append("../")
import matplotlib.image as mpimg
import manager
from manager.manager import Manager
import matplotlib.pyplot as plt
import cv2 
folder_path =  "../dataset/skin_lesion_dataset-master/whole_ground/"
alternative_path = "../dataset/skin_lesion_dataset-master/whole/"
alt = sorted(os.listdir(folder_path))
fol = sorted(os.listdir(alternative_path))

both = list(zip(fol, alt))
print(both)
#%%
difficult_path = "../dataset/skin_lesion_dataset-master/prueba/"
metrics_file = "../dataset/metrics.txt"
total_metrics = 0
totals = 0
zoom_factor = (0.8, 0.8)
manager = Manager(image_path = difficult_path, filter_coefficient = 8, 
                radius = 3, reduce_photo = False, zoom_factors = zoom_factor,
                low_percentile = 1, high_percentile = 97)
#%%
manager.full_stack()
manager.create_borders("X_channel")
manager.plot_metric("dice_score")
#%%
for b in both:
    manager.change_path([alternative_path + b[0], folder_path + b[1]])
    output = manager.full_stack()
    manager.create_borders("B_channel")
    manager.plot_metric("dice_score")
    plt.imshow(manager.X_channel, cmap='gray')
    plt.show()
#%%
print(len(output[0]))
#%%
for key in output[1].keys():
    met_list = []
    for metric in output[1][key].keys():
        if metric == "dice_score":
            met_list.append(output[1][key][metric])
            total_metrics += output[1][key][metric]
            totals += 1
            print(f"{metric}: {output[1][key][metric]}")
    file = open(metrics_file, "a")
    file.write(f"{key}: {max(met_list)}\n")
    file.close()
#%%
import skimage.morphology as morpho
for key, image in output[0].items():
    img = morpho.closing(image["predicted"]["X_channel"], morpho.disk(30))
    plt.imshow(img, cmap='gray')
    plt.show()
#%%

for key in output[0].keys():
    original_image = mpimg.imread(f"{alternative_path}ISIC_{key}.jpg").transpose()/255
    original_image = zoom(original_image, (1, zoom_factor[0], zoom_factor[1]))
    original_mask = output[0][key]["Original"].transpose()
    predicted_mask = output[0][key]["predicted"]["X_channel"]
    predicted_edges = cv2.Canny(predicted_mask, 0, 1)
    original_edges = cv2.Canny(original_mask, 0, 1)
    transposed = []
    # transposed.append(original_image.transpose()[0] - edges.transpose())
    transposed.append(original_image[0] - predicted_edges.transpose() + original_edges.transpose())
    transposed.append(original_image[1] - predicted_edges.transpose() - original_edges.transpose())
    # transposed.append(original_image.transpose()[1])
    transposed.append(original_image[2] - predicted_edges.transpose() - original_edges.transpose())
    # transposed.append(original_image.transpose()[2] - edges.transpose())
    plt.imshow(np.array(transposed).transpose() )
    plt.show()
#%%


for path in both:
    list_both = [alternative_path + path[1]] + [folder_path + path[0]]
    manager.change_path(list_both)
    output = manager.full_stack()
    for key in output[1].keys():
        met_list = []
        for metric in output[1][key].keys():
            if metric == "dice_score":
                met_list.append(output[1][key][metric])
                total_metrics += output[1][key][metric]
                totals += 1
                print(f"{metric}: {output[1][key][metric]}")
        file = open(metrics_file, "a")
        file.write(f"{key}: {max(met_list)}\n")
        file.close()

    for key in output[0].keys():
        original_image = mpimg.imread(f"{alternative_path}ISIC_{key}.jpg").transpose()/255
        original_image = zoom(original_image, (1, zoom_factor[0], zoom_factor[1]))
        original_mask = output[0][key]["Original"].transpose()
        predicted_mask = output[0][key]["predicted"]["X_channel"]
        predicted_edges = cv2.Canny(predicted_mask, 0, 1)
        original_edges = cv2.Canny(original_mask, 0, 1)
        transposed = []
        # transposed.append(original_image.transpose()[0] - edges.transpose())
        transposed.append(original_image[0] - predicted_edges.transpose() + original_edges.transpose())
        transposed.append(original_image[1] - predicted_edges.transpose() - original_edges.transpose())
        # transposed.append(original_image.transpose()[1])
        transposed.append(original_image[2] - predicted_edges.transpose() - original_edges.transpose())
        # transposed.append(original_image.transpose()[2] - edges.transpose())
        plt.imshow(np.array(transposed).transpose() )
        plt.show()

#%%
print(f"The average dice score is: {total_metrics/totals}")
#%%
manager.create_borders("X_channel")
#%%

counter = 0
for key in output[0].keys():
    print("For the image ", key)
    plt.imshow(output[0][key]["Original"].transpose(), cmap='gray')
    plt.show()
    print("You have this mask: ")
    for key, mask in output[0][key]["predicted"].items():
        plt.imshow(mask, cmap='gray')
        plt.show()
    counter +=1 

#%%
# use x_channel
counter = 0
path =  "../dataset/skin_lesion_dataset-master/melanoma"

for key in output[0].keys():
    # original_image = mpimp.i
    original_image = mpimg.imread(f"{path}/ISIC_{key}.jpg").transpose()/255
    original_image = zoom(original_image, (1, 0.1,0.1))
    original_mask = output[0][key]["Original"].transpose()
    predicted_mask = output[0][key]["predicted"]["X_channel"]
    predicted_edges = cv2.Canny(predicted_mask, 0, 1)
    original_edges = cv2.Canny(original_mask, 0, 1)
    transposed = []
    # transposed.append(original_image.transpose()[0] - edges.transpose())
    transposed.append(original_image[0] - predicted_edges.transpose() + original_edges.transpose())
    transposed.append(original_image[1] - predicted_edges.transpose() - original_edges.transpose())
    # transposed.append(original_image.transpose()[1])
    transposed.append(original_image[2] - predicted_edges.transpose() - original_edges.transpose())
    # transposed.append(original_image.transpose()[2] - edges.transpose())
    plt.imshow(np.array(transposed).transpose() )
    plt.show()
# %%
man = Manager(image_path= "../dataset/skin_lesion_dataset-master/prueba")

#%%
channel = mpimg.imread("../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000030.jpg")
grey_hist = np.histogram(channel[0], bins=255, range=(0,255))
#%%
channel.shape
#%%
plt.imshow(channel)
plt.show()

#%%
# plt.imshow(channel[0].transpose(),cmap='grey')
# plt.show()
hist = plt.hist(channel[0])
#%%
plt.hist(channel[1])
plt.show()

#%%
mask = np.where(channel[1]>170, 0, 1)
plt.imshow(mask.transpose())
plt.show()
#%%
manager.change_path(["../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000142.jpg"])
dict_img = manager.predict("../dataset/skin_lesion_dataset-master/melanoma/ISIC_0000142.jpg")

#%%
for key in dict_img.keys():
    plt.imshow(dict_img[key], cmap='gray')
    plt.show()
