import keras.backend as K
import numpy as np

EPS = 1e-12


def get_iou(gt, pr, n_classes):
	class_wise = np.zeros(n_classes)
	for cl in range(n_classes):
		intersection = np.sum((gt == cl) * (pr == cl))
		union = np.sum(np.maximum((gt == cl), (pr == cl)))
		iou = float(intersection) / (union + EPS)
		class_wise[cl] = iou
	return class_wise


def get_f1(y_true, y_pred):  # taken from old keras source code
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
	return f1_val
