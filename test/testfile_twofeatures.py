from __future__ import print_function
import math
import sys
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 35
pd.options.display.max_columns = 35
pd.options.display.float_format = '{:.3f}'.format

music1 = pd.read_csv('music1.csv', sep = ',')

music1 = music1.reindex(np.random.permutation(music1.index))
#print(music1.describe(include = 'all'))
if '-nodfdesc' not in sys.argv:
	print(music1.describe())

def preprocess_features(music1):
	selected_features = music1[["artist.hotttnesss", "duration"]]
	processed_features = selected_features.copy()
	return processed_features

def preprocess_targets(music1):
  	output_targets = pd.DataFrame()
  	output_targets["tempo"] = music1["tempo"]
  	return output_targets

training_examples = preprocess_features(music1.head(7000))
training_targets = preprocess_targets(music1.head(7000))
validation_examples = preprocess_features(music1.tail(3000))
validation_targets = preprocess_targets(music1.tail(3000))

if '-nodesc' not in sys.argv:
	print('\nTraining Examples Summary:')
	print('\n', training_examples.describe())

	print('\nTraining Targets Summary:')
	print('\n', training_targets.describe())

	print('\nValidation Examples Summary:')
	print('\n', validation_examples.describe())

	print('\nValidation Targets Summary:')
	print('\n', validation_targets.describe())

correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["tempo"]

if '-nocorr' not in sys.argv:
	print('\nCorrelation Matrix for Training Examples:')
	print(correlation_dataframe.corr())
	print('\nCorrelation Matrix for Dataset:')
	print(music1.corr())

def construct_feature_columns(input_features):
	return set([tf.feature_column.numeric_column(my_feature)
				for my_feature in input_features])

def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
	features = {key:np.array(value) for key, value in dict(features).items()}

	ds = Dataset.from_tensor_slices((features, targets))
	ds = ds.batch(batch_size).repeat(num_epochs)

	if shuffle:
		ds = ds.shuffle(2500)

	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

def train_model(
	learning_rate,
	steps,
	batch_size,
	training_examples,
	training_targets,
	validation_examples,
	validation_targets):
	periods = 10
	steps_per_period = steps / periods

	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 0.5)
	linear_regressor = tf.estimator.LinearRegressor(
		feature_columns = construct_feature_columns(training_examples),
		optimizer = my_optimizer
	)

	training_input_fn = lambda: my_input_fn(training_examples,
											training_targets["tempo"],
											batch_size = batch_size)

	predict_training_input_fn = lambda: my_input_fn(training_examples,
													training_targets["tempo"],
											num_epochs = 1)

	predict_validation_input_fn = lambda: my_input_fn(validation_examples,
													validation_targets["tempo"],
													num_epochs = 1,
													shuffle = False)

	print('\n--Training Model--')
	print('\nRMSE on Training Data:')
	training_rmse = []
	validation_rmse = []

	for period in range (0, periods):
		linear_regressor.train(
			input_fn = training_input_fn,
			steps = steps_per_period
		)

		training_predictions = linear_regressor.predict(input_fn = predict_training_input_fn)
		training_predictions = np.array([item['predictions'][0] for item in training_predictions])

		validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
		validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

		training_root_mean_squared_error = math.sqrt(
			metrics.mean_squared_error(training_predictions, training_targets))

		validation_root_mean_squared_error = math.sqrt(
			metrics.mean_squared_error(validation_predictions, validation_targets))

		print('Period %02d: %0.2f' % (period, training_root_mean_squared_error))
		training_rmse.append(training_root_mean_squared_error)
		validation_rmse.append(validation_root_mean_squared_error)

	print('\n--Model Training Finished--')

	if '-noplot' not in sys.argv:
		plt.ylabel('RMSE')
		plt.xlabel('Periods')
		plt.title('RMSE v/s Periods')
		plt.tight_layout()
		plt.plot(training_rmse, label = 'Training')
		plt.plot(validation_rmse, label = 'Validation')
		plt.legend()
		plt.show()

	return linear_regressor

minimal_training_examples = training_examples.copy()
minimal_validation_examples = validation_examples.copy()

train_model(
	learning_rate = 0.03,
	steps = 500,
	batch_size = 5,
	training_examples = minimal_training_examples,
	training_targets = training_targets,
	validation_examples = minimal_validation_examples,
	validation_targets = validation_targets)