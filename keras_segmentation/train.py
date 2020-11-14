import glob
import json
import os
from pathlib import Path

import numpy as np
import six
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import init_ops, math_ops

from .data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset
from .metrics import f1_score


def find_latest_checkpoint(checkpoints_path, fail_safe=True):
	def get_epoch_number_from_path(path):
		return path.replace(checkpoints_path, "").strip(".")

	# Get all matching files
	all_checkpoint_files = glob.glob(checkpoints_path + ".*")
	all_checkpoint_files = [ff.replace(".index", "") for ff in
							all_checkpoint_files]  # to make it work for newer versions of keras
	# Filter out entries where the epoc_number part is pure number
	all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
									   .isdigit(), all_checkpoint_files))
	if not len(all_checkpoint_files):
		# The glob list is empty, don't have a checkpoints_path
		if not fail_safe:
			raise ValueError("Checkpoint path {0} invalid"
							 .format(checkpoints_path))
		else:
			return None

	# Find the checkpoint file with the maximum epoch
	latest_epoch_checkpoint = max(all_checkpoint_files,
								  key=lambda f:
								  int(get_epoch_number_from_path(f)))
	return latest_epoch_checkpoint


def masked_categorical_crossentropy(gt, pr):
	from keras.losses import categorical_crossentropy
	mask = 1 - gt[:, :, 0]
	return categorical_crossentropy(gt, pr) * mask


class CheckpointsCallback(Callback):
	def __init__(self, checkpoints_path):
		self.checkpoints_path = checkpoints_path

	def on_epoch_end(self, epoch, logs=None, prefix='ckpt'):
		if self.checkpoints_path is not None:
			path_out = os.path.join(self.checkpoints_path, f'{prefix}.{epoch}')
			self.model.save_weights(path_out)
			print("saved ", path_out)

			for i in range(0, epoch - 2):
				files_path = glob.glob(f'{self.checkpoints_path}/{prefix}.{i}.*')
				for f_path in files_path:
					os.remove(f_path)



def train(
		model, train_images, train_annotations, input_height=None, input_width=None,
		n_classes=None, verify_dataset=True, checkpoints_path=None, epochs=5, batch_size=2,
		validate=False, val_images=None, val_annotations=None, val_batch_size=2,
		auto_resume_checkpoint=False, load_weights=None, steps_per_epoch=512,
		val_steps_per_epoch=512, gen_use_multiprocessing=False, ignore_zero_class=False,
		optimizer_name='adam', do_augment=False, dropout=False, augmentation_name="aug_all",
		loss='categorical_crossentropy', logs_path='../drive/logs', lr=0.0005
):
	print('Model:\t\t\t', model)
	print('Dropout:\t\t', dropout)
	print('Input size:\t\t', f'{input_width}x{input_height} (wxh)')
	print('Num. classes:\t\t', n_classes)
	print('Checkpoint Dir:\t\t', checkpoints_path)
	print('Epochs:\t\t\t', epochs)
	print('Validate:\t\t', validate)
	print('Val. batch size:\t', val_batch_size)
	print('Auto resume checkpoint:\t', auto_resume_checkpoint)
	print('Use Multiprocessing:\t', gen_use_multiprocessing)
	print('Optimizer:\t\t', optimizer_name)
	print('Learning Rate:\t\t', lr)

	from .models.all_models import model_from_name
	# check if user gives model name instead of the model object
	if isinstance(model, six.string_types):
		# create the model from the name
		assert (n_classes is not None), "Please provide the n_classes"
		if (input_height is not None) and (input_width is not None):
			model = model_from_name[model](
				n_classes, input_height=input_height,
				input_width=input_width, dropout=dropout)
		else:
			model = model_from_name[model](n_classes, dropout=dropout)

	n_classes = model.n_classes
	input_height = model.input_height
	input_width = model.input_width
	output_height = model.output_height
	output_width = model.output_width

	if validate:
		assert val_images is not None
		assert val_annotations is not None

	if optimizer_name is not None:

		if ignore_zero_class:
			loss_k = masked_categorical_crossentropy
		else:
			loss_k = loss
		opt = optimizer_name
		if (optimizer_name.lower() == 'adam'):
			opt = tf.keras.optimizers.Adam(learning_rate=lr)
		elif (optimizer_name.lower() == 'sgd'):
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
				initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9
			)
			opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
		elif (optimizer_name.lower() == 'rmsprop'):
			opt = tf.keras.optimizers.RMSprop(
				learning_rate=lr, rho=0.9, momentum=0.0,
				epsilon=1e-07, centered=True, name='RMSprop',
			)

		model.compile(
			loss=loss_k, optimizer=opt,
			metrics=[
				'accuracy',
				tf.keras.metrics.Recall(),
				tf.keras.metrics.Precision(),
				tf.keras.metrics.MeanIoU(num_classes=n_classes),
				f1_score,
				tf.keras.metrics.FalsePositives(name='FP'),
				tf.keras.metrics.TruePositives(name='TP'),
				tf.keras.metrics.FalseNegatives(name='FN'),
				tf.keras.metrics.TrueNegatives(name='TN'),
			]
		)

	if checkpoints_path is not None:
		with open(os.path.join(checkpoints_path, "ckpt_config.json"), "w") as f:
			json.dump({
				"model_class": model.model_name,
				"n_classes": n_classes,
				"input_height": input_height,
				"input_width": input_width,
				"output_height": output_height,
				"output_width": output_width
			}, f)

	if load_weights is not None and len(load_weights) > 0:
		print("Loading weights from ", load_weights)
		model.load_weights(load_weights)

	if auto_resume_checkpoint and (checkpoints_path is not None):
		latest_checkpoint = find_latest_checkpoint(checkpoints_path)
		if latest_checkpoint is not None:
			print("Loading the weights from latest checkpoint ",
				  latest_checkpoint)
			model.load_weights(latest_checkpoint)

	if verify_dataset:
		print("Verifying training dataset")
		verified = verify_segmentation_dataset(train_images,
											   train_annotations,
											   n_classes)
		assert verified
		if validate:
			print("Verifying validation dataset")
			verified = verify_segmentation_dataset(val_images,
												   val_annotations,
												   n_classes)
			assert verified

	train_gen = image_segmentation_generator(
		train_images, train_annotations, batch_size, n_classes,
		input_height, input_width, output_height, output_width,
		do_augment=do_augment, augmentation_name=augmentation_name)

	if validate:
		val_gen = image_segmentation_generator(
			val_images, val_annotations, val_batch_size,
			n_classes, input_height, input_width, output_height, output_width)

	drop = 1 if dropout else 0
	folder_name = f'model-{model.model_name}_opt-{optimizer_name}_loss-{loss}_batch-{batch_size}_epoch-{epochs}_LR-{lr}_dropout-{drop}'
	logdir = f"{logs_path}/{folder_name}"
	Path(logdir).mkdir(parents=True, exist_ok=True)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

	callbacks = [
		CheckpointsCallback(checkpoints_path),
		tensorboard_callback
	]

	if not validate:
		return model.fit_generator(train_gen, steps_per_epoch,
							epochs=epochs, callbacks=callbacks)
	else:
		return model.fit_generator(train_gen,
							steps_per_epoch,
							validation_data=val_gen,
							validation_steps=val_steps_per_epoch,
							epochs=epochs, callbacks=callbacks,
							use_multiprocessing=gen_use_multiprocessing)
