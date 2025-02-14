import sys
sys.path.append("../")
import manager
from manager import Manager
def adjust_hyperparameter_max_last():
    print("Starts")
    best_metrics = {}
    difficult_path = "../dataset/skin_lesion_dataset-master/wholewhole/"
    for last_max in range(52, 83, 10):
        manager = Manager(image_path = difficult_path, filter_coefficient = 8, 
                radius = 3, low_percentile = 1, high_percentile = 97, last_max = last_max)
        output = manager.full_stack()
        for key in output[1].keys():
            best_for_channel = best_metrics.get(key, {})
            dice_score = best_for_channel.get("dice_score", 0)
            if dice_score < output[1][key]["dice_score"]:
                best_metrics[key] = {
                    "dice_score": output[1][key]["dice_score"],
                    "last_max": last_max,
                } 
        dice_score_list = [output[1][i]["dice_score"] for i in output[1].keys()]
        print(
            "dice_score", max(dice_score_list),
            "last_max", last_max,
            )

    print("ends")
    print(best_metrics)

def adjust_hyperparameters_radius_filtercoef_low_high():
    print("Starts")
    difficult_path = "../dataset/skin_lesion_dataset-master/wholewhole/"

    best_metrics = {}
    for radius in [2, 3, 4]:
        for filter_coefficient in range(6, 20, 2):
            for low_percentile in range(1, 3):
                for high_percentile in range(97, 100):
                    manager = Manager(
                        image_path = difficult_path, filter_coefficient=filter_coefficient, radius = radius, 
                        
                        # this are not hyperparameters, are just to decrease the size 
                        reduce_photo = False, zoom_factors = (0.3, 0.3),
                        # of the images and the compute size

                        low_percentile = low_percentile, high_percentile = high_percentile,
                        # last_max = 52,
                        )

                    output = manager.full_stack()
                    for key in output[1].keys():
                        best_for_channel = best_metrics.get(key, {})
                        dice_score = best_for_channel.get("dice_score", 0)
                        if dice_score < output[1][key]["dice_score"]:
                            best_metrics[key] = {
                                "dice_score": output[1][key]["dice_score"],
                                "filter_coefficient": filter_coefficient,
                                "radius": radius,
                                "low_percentile": low_percentile,
                                "high_percentile": high_percentile,
                            } 

                    dice_score_list = [output[1][i]["dice_score"] for i in output[1].keys()]
                    print(
                        "filter_coefficient", filter_coefficient, 
                        "radius", radius, 
                        "dice_score", max(dice_score_list),
                        "low_percentile", low_percentile,
                        "high_percentile", high_percentile,
                        )

    print("ends")
    print(best_metrics)


if __name__ == "__main__":
    adjust_hyperparameters_radius_filtercoef_low_high()



#Results
# {'X_channel': {'dice_score': np.float64(0.8837410643833554), 'contrast_factor': 1.7, 'maximum_filter_size': 10, 'radius': 2}, 'XoYoR_channel': {'dice_score': np.float64(0.8865798124807128), 'contrast_factor': 2.5, 'maximum_filter_size': 10, 'radius': 2}, 'XoYoZoR_channel': {'dice_score': np.float64(0.8860204267878654), 'contrast_factor': 2.5, 'maximum_filter_size': 10, 'radius': 2}, 'R_channel': {'dice_score': np.float64(0.8548009923342365), 'contrast_factor': 2.5, 'maximum_filter_size': 15, 'radius': 2}}
#{'X_channel': {'dice_score': np.float64(0.8523344204743246), 'filter_coefficient': 10, 'radius': 2}, 'XoYoR_channel': {'dice_score': np.float64(0.8562272028580638), 'filter_coefficient': 10, 'radius': 2}, 'XoYoZoR_channel': {'dice_score': np.float64(0.8548892219004149), 'filter_coefficient': 10, 'radius': 2}, 'R_channel': {'dice_score': np.float64(0.6465417271512288), 'filter_coefficient': 10, 'radius': 1}}
# 'X_channel': {'dice_score': np.float64(0.885989433724775), 'filter_coefficient': 8, 'radius': 2}, 'XoYoR_channel': {'dice_score': np.float64(0.8879510674423253), 'filter_coefficient': 8, 'radius': 2}, 'XoYoZoR_channel': {'dice_score': np.float64(0.889970158200542), 'filter_coefficient': 9, 'radius': 2}, 'R_channel': {'dice_score': np.float64(0.8233258827187715), 'filter_coefficient': 7, 'radius': 2}}
# Best one 'filter_coefficient': 8, 'radius': 3, last_max: 52, low_percentile: 1, high_percentile: 97


