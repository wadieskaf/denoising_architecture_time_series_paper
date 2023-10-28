# Import the necessary libraries
import os
import re
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from string import Template
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import code files
from data_generators import DataGenerator, DataGeneratorWLabels
from utils import normalize_data_list_one_per_time, load_json, list_to_string
from lstm_autoencoder import build_lstm_autoencoder, train_lstm_autoencoder, evaluate_lstm_autoencoder

# Hide tensorflow warnings
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Set constants

# $ Random seed
RANDOM_SEED = 42

# $ Environment settings
DEVICE_NAME = '/CPU:0'
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# $ For the data
MAIN_DATASET_PATH = './Datasets/Yahoo S5/ydata-labeled-time-series-anomalies-v1_0'
DATASET_NAMES = ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']
DATASETS_ATTRIBUTES_FILE = './datasets_attributes.json'
DATASETS_ATTRIBUTES = load_json(DATASETS_ATTRIBUTES_FILE)
DATASETS_PATHS = [os.path.join(MAIN_DATASET_PATH, dataset_name) for dataset_name in DATASET_NAMES]
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
NORMALIZE_DATA = True
NORMALIZE_RANGE = (-1, 1)
NORMALIZE_FUNCTION = normalize_data_list_one_per_time
SHUFFLE_DATA = True

# $ For Model building and training
EXPERIMENTS_FILE_PATH = './exps.json'
LOSS_FUNCTION = 'mae'
OPTIMIZER = 'adam'
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
EPOCHS = 500
PATIENCE = 3
TRAINING_VERBOSE = 2
MODEL_CHECKPOINT_VERBOSE = 2
EARLY_STOPPING_VERBOSE = 2

# $ For saving models and results
FILE_NAME_TEMPLATE = Template('${dataset_name}_${architecture}_${seq_len}_(${dropout})')
MAIN_FOLDER_PATH = './experiments_results/'
BEST_MODELS_FOLDER = os.path.join(MAIN_FOLDER_PATH, 'best_models')
FINAL_MODELS_FOLDER = os.path.join(MAIN_FOLDER_PATH, 'final_models')
RESULTS_FOLDER = os.path.join(MAIN_FOLDER_PATH, 'results')
CONFUSION_MATRICES_FOLDER = os.path.join(MAIN_FOLDER_PATH, 'confusion_matrices')
HISTORY_FOLDER = os.path.join(MAIN_FOLDER_PATH, 'history')

# create the folders if they don't exist
for folder in [MAIN_FOLDER_PATH, BEST_MODELS_FOLDER, FINAL_MODELS_FOLDER, RESULTS_FOLDER, CONFUSION_MATRICES_FOLDER, HISTORY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

with tf.device(DEVICE_NAME):
    df_results = pd.DataFrame(columns=['dataset', 'architecture', 'seq_len', 'dropout', 'epochs', 'final_train_loss',
                                       'final_valid_loss', 'best_valid_loss', 'best_epoch',
                                       'precision', 'recall', 'f1_score', 'accuracy'])

    experiments = load_json(EXPERIMENTS_FILE_PATH)

    for dataset_name, dataset_path in zip(DATASET_NAMES, DATASETS_PATHS):
        # Get the dataset attributes
        dataset_attributes = DATASETS_ATTRIBUTES[dataset_name]
        value_column = dataset_attributes['value_column']
        label_column = dataset_attributes['label_column']
        file_name_pattern = re.compile(dataset_attributes['file_name_pattern'])
        # Get the dataset files
        dataset_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if
                         file_name_pattern.match(file)]
        # load the dataset
        files_dfs = [pd.read_csv(dataset_file) for dataset_file in dataset_files]
        # get values and labels
        dataset_values = [df[value_column].values for df in files_dfs]
        dataset_labels = [df[label_column].values for df in files_dfs]
        # split the dataset into train, valid and test
        train_data = [d[:int(TRAIN_SIZE * len(d))] for d in dataset_values]
        valid_data = [d[int(TRAIN_SIZE * len(d)):int((TRAIN_SIZE + VALID_SIZE) * len(d))] for d in dataset_values]
        test_data = [d[int((TRAIN_SIZE + VALID_SIZE) * len(d)):] for d in dataset_values]
        test_labels = [d[int((TRAIN_SIZE + VALID_SIZE) * len(d)):] for d in dataset_labels]
        for experiment in experiments.values():
            architecture = experiment['architecture']
            seq_lens = experiment['seq_lens']
            dropouts = experiment['dropouts']
            for seq_len in seq_lens:
                for dropout in dropouts:
                    print(f'######## STARTING {dataset_name}_{architecture}_{seq_len}_{dropout} EXPERIMENT ########')
                    # create file name template for the current experiment
                    file_name_template = FILE_NAME_TEMPLATE.substitute(dataset_name=dataset_name,
                                                                       architecture=architecture, dropout=dropout,
                                                                       seq_len=seq_len)
                    # Initiate the data generators
                    train_gen = DataGenerator(train_data, batch_size=TRAIN_BATCH_SIZE, seq_len=seq_len,
                                              normalize=NORMALIZE_DATA,
                                              normalize_range=NORMALIZE_RANGE, normalize_function=NORMALIZE_FUNCTION,
                                              test=True, shuffle_data=SHUFFLE_DATA, random_seed=RANDOM_SEED)
                    valid_gen = DataGenerator(valid_data, batch_size=VALID_BATCH_SIZE, seq_len=seq_len,
                                              normalize=NORMALIZE_DATA,
                                              normalize_range=NORMALIZE_RANGE, normalize_function=NORMALIZE_FUNCTION,
                                              test=True)
                    test_gen = DataGeneratorWLabels(test_data, test_labels, batch_size=TEST_BATCH_SIZE, seq_len=seq_len,
                                                    normalize=NORMALIZE_DATA, normalize_range=NORMALIZE_RANGE,
                                                    normalize_function=NORMALIZE_FUNCTION, test=True)

                    # Build the model
                    ae = build_lstm_autoencoder(input_shape=(TRAIN_BATCH_SIZE, seq_len, 1), layers=architecture,
                                                dropout=dropout,
                                                loss_function=LOSS_FUNCTION, optimizer=OPTIMIZER)
                    print(ae.summary())
                    # prepare the callbacks
                    save_best = ModelCheckpoint(
                        os.path.join(BEST_MODELS_FOLDER, f'best_{file_name_template}.h5'), monitor='val_loss',
                        verbose=MODEL_CHECKPOINT_VERBOSE, save_best_only=True, save_weights_only=False, mode='auto',
                        period=1)
                    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=EARLY_STOPPING_VERBOSE,
                                                   patience=PATIENCE)
                    callbacks = [save_best, early_stopping]
                    # train the model
                    model, history = train_lstm_autoencoder(ae, train_gen, valid_gen, callbacks=callbacks,
                                                            epochs=EPOCHS,
                                                            verbose=TRAINING_VERBOSE)
                    # save the model
                    model.save(os.path.join(FINAL_MODELS_FOLDER, f'{file_name_template}.h5'))
                    # evaluate the model
                    metrics, cm = evaluate_lstm_autoencoder(model, test_gen)
                    # get experiment results
                    history_loss = history.history['loss']
                    history_val_loss = history.history['val_loss']
                    best_val_loss = min(history_val_loss)
                    experiment_results = {
                        'dataset': dataset_name,
                        'architecture': list_to_string(architecture),
                        'seq_len': seq_len,
                        'dropout': dropout,
                        'epochs': len(history_loss),
                        'final_train_loss': history_loss[-1],
                        'final_valid_loss': history_val_loss[-1],
                        'best_epoch': history_val_loss.index(best_val_loss) + 1,
                        'best_valid_loss': best_val_loss,
                    }
                    experiment_results.update(metrics)
                    # print the experiment results
                    df_experiment = pd.DataFrame(experiment_results, index=[0])
                    print(f'######## {dataset_name}_{architecture}_{seq_len}_{dropout} EXPERIMENT RESULTS ########')
                    print(df_experiment.to_string())
                    print('############################################################################')
                    # save the experiment results
                    df_experiment.to_csv(os.path.join(RESULTS_FOLDER, f'{file_name_template}.csv'),
                                         index=False)
                    # add the experiment results to the overall results
                    df_results = df_results.append(df_experiment, ignore_index=True)
                    # save the confusion matrix as np array
                    np.save(os.path.join(CONFUSION_MATRICES_FOLDER, f'{file_name_template}_cm.npy'), cm)
                    # save the history
                    np.save(os.path.join(HISTORY_FOLDER, f'{file_name_template}_history.npy'), history.history)

    # save the overall results
    df_results.to_csv(os.path.join(RESULTS_FOLDER, 'overall.csv'), index=False)
