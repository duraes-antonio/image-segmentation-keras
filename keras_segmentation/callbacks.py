import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from scikitplot.metrics import plot_roc


class ROCCallback(Callback):
	def __init__(self, model, validation_data, image_dir):
		super().__init__()
		self.model = model
		self.validation_data = next(validation_data)
		os.makedirs(image_dir, exist_ok=True)
		self.image_dir = image_dir

	def on_epoch_end(self, epoch, logs={}):
		y_pred = np.asarray(self.model.predict(self.validation_data[0]))
		y_true = self.validation_data[1]

		# plot and save roc curve
		fig, ax = plt.subplots(figsize=(16, 12))
		plot_roc(y_true, y_pred.astype(int), ax=ax)
		fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))