import numpy as np

class Metrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This class is used to calculate the metrics of a model.
        For it to work, the y_true and y_pred should be numpy arrays.
        Then multiple metrics can be calculated.
        The class expects that the y_true and y_pred have the same shape and are both 0-1 masks.
        """
        self.y_true = y_true
        self.y_pred = y_pred
    
    def dice_score(self):
        """
        This function calculates the dice score of the prediction.
        """
        return 2 * np.sum(self.y_true * self.y_pred) / (np.sum(self.y_true) + np.sum(self.y_pred))
    
    def jaccard_score(self):
        """
        This function calculates the jaccard score of the prediction.
        """
        return np.sum(self.y_true * self.y_pred) / np.logical_or(self.y_true, self.y_pred).astype(int).sum()
    
    def recall(self):
        """
        This function calculates the recall of the prediction.
        """
        return np.sum(self.y_true * self.y_pred) / np.sum(self.y_true)
    
    def precision(self):
        """
        This function calculates the precision of the prediction.
        """
        return np.sum(self.y_true * self.y_pred) / np.sum(self.y_pred)
    
    def f1_score(self):
        """
        This function calculates the f1 score of the prediction.
        """
        pre = self.precision()
        rec = self.recall()
        return 2 * (pre * rec) / (pre + rec)
    
    def redefine_sample(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This function redefines the sample to calculate the metrics.
        """
        self.y_true = y_true
        self.y_pred = y_pred
    