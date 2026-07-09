# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:58:36 2025

Deep Learning for tACS artifacts removal

@author: xingl
"""

#%%
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim

from scipy.signal import welch
from scipy.stats import pearsonr

file = r"D:\MyWork\EEGsystem\8通道脑电+电刺激\data\EEG+tACS\EEG_tES\on_phantom_data\EEG_TACS_ALPHA_preprocessed.mat"
mat = scipy.io.loadmat(file)
data = mat['preprocessed_data_Alpha']

EEG_ground_truth_alpha = data[0][0][0][1]
Test_1mA_5Hz_alpha = data[1][0][0][1]
Test_1mA_10Hz_alpha = data[2][0][0][1]
Test_1mA_40Hz_alpha = data[3][0][0][1]
Test_250uA_5Hz_alpha = data[4][0][0][1]
Test_250uA_10Hz_alpha = data[5][0][0][1]
Test_250uA_40Hz_alpha = data[6][0][0][1]

file_erp = r"D:\MyWork\EEGsystem\8通道脑电+电刺激\data\EEG+tACS\EEG_tES\on_phantom_data\EEG_TACS_ERP_preprocessed.mat"
mat_erp = scipy.io.loadmat(file_erp)
data_erp = mat_erp['preprocessed_data_ERP']

EEG_ground_truth_erp = data_erp[0][0][0][1]
Test_1mA_5Hz_erp = data_erp[1][0][0][1]
Test_1mA_10Hz_erp = data_erp[2][0][0][1]
Test_1mA_40Hz_erp = data_erp[3][0][0][1]
Test_250uA_5Hz_erp = data_erp[4][0][0][1]
Test_250uA_10Hz_erp = data_erp[5][0][0][1]
Test_250uA_40Hz_erp = data_erp[6][0][0][1]


#%% Hyperparameters
# ============================================================
# You only need to change the values in this block for most experiments.
# The rest of the script will automatically use these settings.
# ============================================================

# ---------- Data segmentation ----------
sampling_rate = 500
tacs_start_alpha = 27.9 + 1      # tACS onset for Alpha data, in seconds
tacs_start_erp = 26.5 + 1        # tACS onset for ERP data, in seconds
segment_length_seconds = 2       # Change to 1, 2, or 4 for window-length experiments
overlap = 0.5                    # Change to 0, 0.2, or 0.5 for overlap experiments
segment_length_samples = int(segment_length_seconds * sampling_rate)

# ---------- Channel selection ----------
# Your loaded data arrays have shape [3, n_samples].
# Python channel indices are zero-based:
#   [0]       = use only the first row/channel
#   [1]       = use only the second row/channel
#   [2]       = use only the third row/channel
#   [0,1,2]   = pool all three channels as independent single-channel epochs
# If these three rows correspond to manuscript channels Ch1, Ch7, and Ch8,
# keep the labels below as [1, 7, 8].
SELECTED_CHANNEL_INDICES = [0,1,2]      # Default: only use the first channel
SELECTED_CHANNEL_LABELS = [1,7,8]       # Label shown in metadata and printed summaries

if len(SELECTED_CHANNEL_INDICES) != len(SELECTED_CHANNEL_LABELS):
    raise ValueError("SELECTED_CHANNEL_INDICES and SELECTED_CHANNEL_LABELS must have the same length.")

CHANNEL_TAG = "Ch" + "-".join([str(ch) for ch in SELECTED_CHANNEL_LABELS])

# ---------- Train / validation / test split ----------
RANDOM_STATE = 1
TEST_SIZE = 0.2                  # First split: 80% train, 20% temporary
VALID_SIZE_IN_TEMP = 0.5         # Second split: temporary set is split into 50% valid and 50% test

# ---------- Model hyperparameters ----------
USE_BATCH_NORM = True            # Set False for No-BatchNorm ablation
USE_DROPOUT = True               # Set False for No-Dropout ablation
DROPOUT_RATE = 0.2
KERNEL_SIZE = 3

# ---------- Training hyperparameters ----------
EPOCHS = 1000                    # Use a small value, e.g., 10, for a quick test run
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# ---------- Run name / checkpoint ----------
RUN_TAG = (
    f"{CHANNEL_TAG}_"
    f"win{segment_length_seconds}s_"
    f"overlap{int(overlap * 100)}pct_"
    f"BN{int(USE_BATCH_NORM)}_"
    f"Drop{int(USE_DROPOUT)}"
)
CHECKPOINT_PATH = f"best_autoencoder_{RUN_TAG}"

print("\n========== RUN CONFIG ==========")
print("RUN_TAG:", RUN_TAG)
print("sampling_rate:", sampling_rate)
print("segment_length_seconds:", segment_length_seconds)
print("segment_length_samples:", segment_length_samples)
print("overlap:", overlap)
print("SELECTED_CHANNEL_INDICES:", SELECTED_CHANNEL_INDICES)
print("SELECTED_CHANNEL_LABELS:", SELECTED_CHANNEL_LABELS)
print("USE_BATCH_NORM:", USE_BATCH_NORM)
print("USE_DROPOUT:", USE_DROPOUT)
print("DROPOUT_RATE:", DROPOUT_RATE)
print("EPOCHS:", EPOCHS)
print("BATCH_SIZE:", BATCH_SIZE)
print("LEARNING_RATE:", LEARNING_RATE)


#%% data load check
#### Note: 电刺激从27.9s开始 (Alpha Data) ####
# Function to segment signals for multiple input signals (2D arrays)
def segment_signals(tacs_start, clean_signal, corrupted_signal, sampling_rate,
                    segment_length_seconds, overlap,
                    selected_channel_indices=None):
    """
    Segment clean/corrupted signal pairs.

    Important:
    - clean_signal and corrupted_signal are expected to have shape [n_channels, n_samples].
    - selected_channel_indices controls which rows/channels are used.
    - Each selected channel is segmented into independent single-channel epochs.
    """
    segment_length_samples = int(segment_length_seconds * sampling_rate)
    step_size = int(segment_length_samples * (1 - overlap))
    start_samples = int(tacs_start * sampling_rate)
    end_samples = start_samples + 58 * sampling_rate  # Adjust the end point for segmentation
    all_segments = []

    if step_size <= 0:
        raise ValueError("overlap must be < 1. Current overlap gives step_size <= 0.")

    if selected_channel_indices is None:
        selected_channel_indices = list(range(clean_signal.shape[0]))

    # Safety check
    for signal_index in selected_channel_indices:
        if signal_index < 0 or signal_index >= clean_signal.shape[0]:
            raise IndexError(
                f"Selected channel index {signal_index} is out of range for data with "
                f"{clean_signal.shape[0]} channels."
            )

    for signal_index in selected_channel_indices:
        clean_row = clean_signal[signal_index]
        corrupted_row = corrupted_signal[signal_index]
        segments = []

        for start in range(start_samples, end_samples + 1, step_size):
            end = start + segment_length_samples
            if end <= len(clean_row):
                clean_segment = clean_row[start:end]
                corrupted_segment = corrupted_row[start:end]
                segments.append([clean_segment, corrupted_segment])
            else:
                break

        if len(segments) > 0:
            all_segments.append(np.array(segments))

    if len(all_segments) == 0:
        raise ValueError("No segments were created. Check tacs_start, signal length, window length, and channel selection.")

    # Concatenate epochs across selected channels.
    all_segments = np.concatenate(all_segments, axis=0)
    return all_segments

# Generate metadata for each epoch returned by segment_signals().
def make_metadata_for_segments(tacs_start, clean_signal, sampling_rate,
                               segment_length_seconds, overlap,
                               paradigm, amplitude, frequency_hz,
                               selected_channel_indices=None,
                               selected_channel_labels=None):
    """
    Generate metadata for each epoch produced by segment_signals().

    The row order of the returned metadata is designed to match segment_signals():
    selected channel by selected channel, epoch by epoch.

    Therefore:
        metadata.iloc[i] corresponds to segments[i]
    """
    segment_length_samples = int(segment_length_seconds * sampling_rate)
    step_size = int(segment_length_samples * (1 - overlap))
    start_samples = int(tacs_start * sampling_rate)
    end_samples = start_samples + 58 * sampling_rate

    if step_size <= 0:
        raise ValueError("overlap must be < 1. Current overlap gives step_size <= 0.")

    if selected_channel_indices is None:
        selected_channel_indices = list(range(clean_signal.shape[0]))

    if selected_channel_labels is None:
        # Default labels are one-based channel numbers.
        selected_channel_labels = [idx + 1 for idx in selected_channel_indices]

    if len(selected_channel_indices) != len(selected_channel_labels):
        raise ValueError("selected_channel_indices and selected_channel_labels must have the same length.")

    # Safety check
    for signal_index in selected_channel_indices:
        if signal_index < 0 or signal_index >= clean_signal.shape[0]:
            raise IndexError(
                f"Selected channel index {signal_index} is out of range for data with "
                f"{clean_signal.shape[0]} channels."
            )

    metadata_rows = []

    for k, signal_index in enumerate(selected_channel_indices):
        channel_label = selected_channel_labels[k]
        epoch_id = 0
        clean_row = clean_signal[signal_index]

        for start in range(start_samples, end_samples + 1, step_size):
            end = start + segment_length_samples

            if end <= len(clean_row):
                metadata_rows.append({
                    "paradigm": paradigm,                  # Alpha or ERP
                    "amplitude": amplitude,                # 1mA or 250uA
                    "frequency_hz": frequency_hz,          # 5, 10, or 40
                    "channel": channel_label,              # channel label used in reporting
                    "signal_index": signal_index,          # zero-based row index in the data array
                    "epoch_id": epoch_id,                  # epoch number within this channel/condition
                    "start_sample": start,
                    "end_sample": end,
                    "start_time_sec": start / sampling_rate,
                    "end_time_sec": end / sampling_rate,
                    "window_sec": segment_length_seconds,
                    "overlap": overlap,
                    "condition": f"{paradigm}_{amplitude}_{frequency_hz}Hz_Ch{channel_label}"
                })
                epoch_id += 1
            else:
                break

    return pd.DataFrame(metadata_rows)


# Normalize each segment (clean and corrupted) in the segments array using max-min normalization.
def normalize_segments(segments):
    normalized_segments = np.copy(segments)  # Create a copy to avoid modifying the original data
    # Iterate through each segment pair
    for i in range(segments.shape[0]):  # Loop over the number of segments
        for j in range(2):  # Loop over the two types (clean and corrupted)
            segment = segments[i, j]
            
            # Apply max-min normalization to each segment
            min_val = np.min(segment)
            max_val = np.max(segment)
            if max_val != min_val:  # Avoid division by zero
                normalized_segments[i, j] = (segment - min_val) / (max_val - min_val)
            else:
                normalized_segments[i, j] = np.zeros_like(segment)  # In case all values are the same, normalize to zero
                
    return normalized_segments



#%% Data Segmentation
# Segmentation parameters are defined in the Hyperparameters block above.

# alpha data
segments1 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_1mA_5Hz_alpha, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments2 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_1mA_10Hz_alpha, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments3 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_1mA_40Hz_alpha, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments4 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_250uA_5Hz_alpha, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments5 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_250uA_10Hz_alpha, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments6 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_250uA_40Hz_alpha, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)

## erp data
segments11 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_1mA_5Hz_erp, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments22 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_1mA_10Hz_erp, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments33 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_1mA_40Hz_erp, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments44 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_250uA_5Hz_erp, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments55 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_250uA_10Hz_erp, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)
segments66 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_250uA_40Hz_erp, sampling_rate, segment_length_seconds, overlap,
                            selected_channel_indices=SELECTED_CHANNEL_INDICES)


#%% Metadata for each segmented epoch
# These metadata tables are aligned with segments1, segments2, ..., segments66.
# For example, metadata1.iloc[i] corresponds to segments1[i].

metadata1 = make_metadata_for_segments(tacs_start_alpha, EEG_ground_truth_alpha, sampling_rate,
                                       segment_length_seconds, overlap,
                                       paradigm="Alpha", amplitude="1mA", frequency_hz=5,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata2 = make_metadata_for_segments(tacs_start_alpha, EEG_ground_truth_alpha, sampling_rate,
                                       segment_length_seconds, overlap,
                                       paradigm="Alpha", amplitude="1mA", frequency_hz=10,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata3 = make_metadata_for_segments(tacs_start_alpha, EEG_ground_truth_alpha, sampling_rate,
                                       segment_length_seconds, overlap,
                                       paradigm="Alpha", amplitude="1mA", frequency_hz=40,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata4 = make_metadata_for_segments(tacs_start_alpha, EEG_ground_truth_alpha, sampling_rate,
                                       segment_length_seconds, overlap,
                                       paradigm="Alpha", amplitude="250uA", frequency_hz=5,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata5 = make_metadata_for_segments(tacs_start_alpha, EEG_ground_truth_alpha, sampling_rate,
                                       segment_length_seconds, overlap,
                                       paradigm="Alpha", amplitude="250uA", frequency_hz=10,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata6 = make_metadata_for_segments(tacs_start_alpha, EEG_ground_truth_alpha, sampling_rate,
                                       segment_length_seconds, overlap,
                                       paradigm="Alpha", amplitude="250uA", frequency_hz=40,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata11 = make_metadata_for_segments(tacs_start_erp, EEG_ground_truth_erp, sampling_rate,
                                        segment_length_seconds, overlap,
                                        paradigm="ERP", amplitude="1mA", frequency_hz=5,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata22 = make_metadata_for_segments(tacs_start_erp, EEG_ground_truth_erp, sampling_rate,
                                        segment_length_seconds, overlap,
                                        paradigm="ERP", amplitude="1mA", frequency_hz=10,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata33 = make_metadata_for_segments(tacs_start_erp, EEG_ground_truth_erp, sampling_rate,
                                        segment_length_seconds, overlap,
                                        paradigm="ERP", amplitude="1mA", frequency_hz=40,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata44 = make_metadata_for_segments(tacs_start_erp, EEG_ground_truth_erp, sampling_rate,
                                        segment_length_seconds, overlap,
                                        paradigm="ERP", amplitude="250uA", frequency_hz=5,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata55 = make_metadata_for_segments(tacs_start_erp, EEG_ground_truth_erp, sampling_rate,
                                        segment_length_seconds, overlap,
                                        paradigm="ERP", amplitude="250uA", frequency_hz=10,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)

metadata66 = make_metadata_for_segments(tacs_start_erp, EEG_ground_truth_erp, sampling_rate,
                                        segment_length_seconds, overlap,
                                        paradigm="ERP", amplitude="250uA", frequency_hz=40,
                                       selected_channel_indices=SELECTED_CHANNEL_INDICES,
                                       selected_channel_labels=SELECTED_CHANNEL_LABELS)


#%% Data Normalization
normalized_segments1 = normalize_segments(segments1)
normalized_segments2 = normalize_segments(segments2)
normalized_segments3 = normalize_segments(segments3)
normalized_segments4 = normalize_segments(segments4)
normalized_segments5 = normalize_segments(segments5)
normalized_segments6 = normalize_segments(segments6)


normalized_segments11 = normalize_segments(segments11)
normalized_segments22 = normalize_segments(segments22)
normalized_segments33 = normalize_segments(segments33)
normalized_segments44 = normalize_segments(segments44)
normalized_segments55 = normalize_segments(segments55)
normalized_segments66 = normalize_segments(segments66)
    

all_segments = np.concatenate((normalized_segments1,normalized_segments2,normalized_segments3,normalized_segments4,normalized_segments5,normalized_segments6,
                               normalized_segments11,normalized_segments22,normalized_segments33,normalized_segments44,normalized_segments55,normalized_segments66), axis=0)

# Concatenate metadata in exactly the same order as all_segments.
# Therefore, all_metadata.iloc[i] corresponds to all_segments[i].
all_metadata = pd.concat((metadata1, metadata2, metadata3, metadata4, metadata5, metadata6,
                          metadata11, metadata22, metadata33, metadata44, metadata55, metadata66),
                         axis=0).reset_index(drop=True)

# Basic alignment checks
print("all_segments shape:", all_segments.shape)
print("all_metadata shape:", all_metadata.shape)
assert all_segments.shape[0] == len(all_metadata), "ERROR: all_segments and all_metadata length mismatch!"

print("\nMetadata preview:")
print(all_metadata.head())

print("\nEpoch counts by paradigm / amplitude / frequency / channel:")
print(all_metadata.groupby(["paradigm", "amplitude", "frequency_hz", "channel"]).size())

# Optional: save metadata for all epochs
all_metadata.to_csv(f"all_epoch_metadata_{RUN_TAG}.csv", index=False)
    
    


#%% Deep learning model 
######  Define a convolutional Autoencoder
import tensorflow as tf
from tensorflow.keras import layers, Model, losses
from tensorflow import keras

class AutoencoderX(Model):
    def __init__(self, input_length=1000, kernel_size=3,
                 use_batch_norm=True, use_dropout=True, dropout_rate=0.2):
        super(AutoencoderX, self).__init__()

        self.input_length = input_length
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        def conv_block(filters, strides=1, transpose=False):
            block = []
            if transpose:
                block.append(layers.Conv1DTranspose(filters, kernel_size,
                                                    activation=None,
                                                    padding='same',
                                                    strides=strides))
            else:
                block.append(layers.Conv1D(filters, kernel_size,
                                           activation=None,
                                           padding='same',
                                           strides=strides))
            if use_batch_norm:
                block.append(layers.BatchNormalization())
            block.append(layers.LeakyReLU(alpha=0.1))
            return block

        encoder_layers = [layers.Input(shape=(input_length, 1))]
        encoder_layers += conv_block(64, strides=1)
        encoder_layers += conv_block(32, strides=1)
        encoder_layers += conv_block(16, strides=2)
        encoder_layers += conv_block(8, strides=2)
        encoder_layers += conv_block(4, strides=1)

        if use_dropout:
            encoder_layers.append(layers.Dropout(dropout_rate))

        self.encoder = keras.Sequential(encoder_layers)

        decoder_layers = []
        decoder_layers += conv_block(8, strides=1)
        decoder_layers += conv_block(16, strides=2, transpose=True)
        decoder_layers += conv_block(32, strides=2, transpose=True)
        decoder_layers += conv_block(64, strides=1)
        decoder_layers.append(layers.Conv1D(1, kernel_size=kernel_size,
                                            activation='sigmoid',
                                            padding='same'))

        self.decoder = keras.Sequential(decoder_layers)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



#%% Build model and test
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses

# Split indices first, so metadata can be split consistently with all_segments.
# This keeps the original random split logic: test_size=0.2, then 50/50 valid/test from the temporary set.
all_indices = np.arange(all_segments.shape[0])

train_idx, tem_idx = train_test_split(all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
valid_idx, test_idx = train_test_split(tem_idx, test_size=VALID_SIZE_IN_TEMP, random_state=RANDOM_STATE, shuffle=True)

train_data = all_segments[train_idx]
valid_data = all_segments[valid_idx]
test_data = all_segments[test_idx]

train_metadata = all_metadata.iloc[train_idx].reset_index(drop=True)
valid_metadata = all_metadata.iloc[valid_idx].reset_index(drop=True)
test_metadata = all_metadata.iloc[test_idx].reset_index(drop=True)

print("\nTrain / Valid / Test metadata check:")
print("train_data:", train_data.shape, "train_metadata:", train_metadata.shape)
print("valid_data:", valid_data.shape, "valid_metadata:", valid_metadata.shape)
print("test_data:", test_data.shape, "test_metadata:", test_metadata.shape)

assert train_data.shape[0] == len(train_metadata)
assert valid_data.shape[0] == len(valid_metadata)
assert test_data.shape[0] == len(test_metadata)

print("\nTest set condition counts:")
print(test_metadata.groupby(["paradigm", "amplitude", "frequency_hz", "channel"]).size())

# Optional: save split metadata
train_metadata.to_csv(f"train_metadata_{RUN_TAG}.csv", index=False)
valid_metadata.to_csv(f"valid_metadata_{RUN_TAG}.csv", index=False)
test_metadata.to_csv(f"test_metadata_{RUN_TAG}.csv", index=False)

train_corrupted = train_data[:,1,:]
train_clean = train_data[:,0,:]
valid_corrupted = valid_data[:,1,:]
valid_clean = valid_data[:,0,:]
test_corrupted = test_data[:,1,:]
test_clean = test_data[:,0,:]

train_corrupted = np.expand_dims(train_corrupted, -1)
train_clean = np.expand_dims(train_clean, -1)
valid_corrupted = np.expand_dims(valid_corrupted, -1)
valid_clean = np.expand_dims(valid_clean, -1)
test_corrupted = np.expand_dims(test_corrupted, -1)
test_clean = np.expand_dims(test_clean, -1)


### check GPU is ok
gpu_device_name = tf.test.gpu_device_name()
print('GPU:', tf.test.is_gpu_available())
print(gpu_device_name)


# Define checkpoint callback (use SavedModel format)
checkpoint_callback = ModelCheckpoint(
    # filepath="best_autoencoder",  # No .h5 extension → TensorFlow SavedModel format
    filepath=CHECKPOINT_PATH,  # No extension → TensorFlow SavedModel format
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
    save_format="tf"  # Save using TensorFlow format instead of HDF5
)


# Compile the autoencoder model
autoencoder = AutoencoderX(
    input_length=segment_length_samples,
    kernel_size=KERNEL_SIZE,
    use_batch_norm=USE_BATCH_NORM,
    use_dropout=USE_DROPOUT,
    dropout_rate=DROPOUT_RATE
)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

autoencoder.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=['accuracy'])

# Train the model
history = autoencoder.fit(
    train_corrupted, train_clean, 
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True, 
    validation_data=(valid_corrupted, valid_clean),
    callbacks=[checkpoint_callback]
)

# Load the best model
# best_autoencoder = tf.keras.models.load_model("best_autoencoder")
# best_autoencoder = tf.keras.models.load_model("best_autoencoder_20overlap")

# Model details
autoencoder.encoder.summary()
autoencoder.decoder.summary()


encoded_layer = autoencoder.encoder(test_corrupted).numpy()
decoded_layer = autoencoder.decoder(encoded_layer).numpy()
decoded_layer = np.squeeze(decoded_layer) # back to 2-dimensional array


# Example: check the source information of any test epoch.
# test_metadata.iloc[i] corresponds to test_clean[i], test_corrupted[i], and decoded_layer[i].
# i = 0
# print(test_metadata.iloc[i])



## save model
# autoencoder.save("saved_model/autoencoderX.h5")  
# autoencoder.save('saved_model/AutoencoderX.keras')
# # Convert the model to TensorFlow Lite and save
# converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
# tflite_model = converter.convert()
# with open('saved_model/AutoencoderX.tflite', 'wb') as f:
#   f.write(tflite_model)



#%%
fs = sampling_rate
time = np.linspace(0, len(test_clean[0])/fs, num=len(test_clean[0]))
n = 20  # We want to display 20 pairs of signals
plt.figure(figsize=(30, 10))
for i in range(n):
    # Create a subplot for each pair of signals
    ax = plt.subplot(2, 10, i + 1)  # 4 rows, 10 columns
    plt.title(f"Signal {i+1}")
    # Plot test_clean in blue
    plt.plot(time, test_clean[i, :], label="Original", color='b')
    # Plot decoded_layer in red
    plt.plot(time, decoded_layer[i, :], label="Reconstructed", color='r')
    # Show the legend
    plt.legend()
    # Make sure the axes are visible
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

plt.tight_layout()  # Adjust layout for better spacing between subplots
plt.show()


#%% evaluation
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import welch
import math

def evaluate_model_performance(test_clean, reconstructed_data, fs=500):
    # Ensure the data is 2D (samples x time points)
    if test_clean.ndim == 3:
        test_clean = test_clean.squeeze()
        reconstructed_data = reconstructed_data.squeeze()

    # Time Domain Evaluation
    # Calculate RRMSE (Relative Root Mean Squared Error)
    def rmsValue(arr):
        square = 0
        mean = 0.0
        root = 0.0
        n = len(arr)
        #Calculate square
        for i in range(0,n):
            square += (arr[i]**2)
        #Calculate Mean
        mean = (square / (float)(n))
        #Calculate Root
        root = math.sqrt(mean)
        return root

    def rrmse(true, pred):
        num = rmsValue(true-pred)
        den = rmsValue(true)
        rrmse_loss = num/den
        return rrmse_loss

    # Calculate Correlation Coefficient (CC) in Time Domain
    def correlation(x, y):
        return pearsonr(x.flatten(), y.flatten())[0]  # Flatten to 1D for CC calculation

    rrmse_time_results = []
    cc_time_results = []

    # Frequency Domain Evaluation (Using Welch Transform)
    # Compute Power Spectral Density (PSD) using Welch's method
    NPERSEG = 200

    rrmse_freq_results = []
    cc_freq_results = []

    for i in range(test_clean.shape[0]):
        # Get the clean, corrupted, and reconstructed data for this sample
        clean_signal = test_clean[i]
        reconstructed_signal = reconstructed_data[i]

        # Calculate Time Domain Metrics
        rrmse_time_results.append(rrmse(clean_signal, reconstructed_signal))
        cc_time_results.append(correlation(clean_signal, reconstructed_signal))

        # Calculate Frequency Domain Metrics
        f, psd_clean = welch(clean_signal, fs=fs, nperseg=NPERSEG)
        _, psd_reconstructed = welch(reconstructed_signal, fs=fs, nperseg=NPERSEG)

        # RRMSE for PSD
        rrmse_freq_results.append(rrmse(psd_clean, psd_reconstructed))

        # Correlation for PSD
        cc_freq_results.append(correlation(psd_clean, psd_reconstructed))

    # Calculate averages and standard deviations for time and frequency domain results
    def calculate_stats(results):
        avg = np.mean(results)
        std_dev = np.std(results)
        return avg, std_dev

    time_avg = {
        'RRMSE': calculate_stats(rrmse_time_results),
        'CC': calculate_stats(cc_time_results)
    }

    freq_avg = {
        'RRMSE': calculate_stats(rrmse_freq_results),
        'CC': calculate_stats(cc_freq_results)
    }

    # Print the results in a readable format (each on a new line)
    print("Time Domain Evaluation:")
    print(f"RRMSE: {time_avg['RRMSE'][0]:.4f} ± {time_avg['RRMSE'][1]:.4f}")
    print(f"Correlation Coefficient (CC): {time_avg['CC'][0]:.4f} ± {time_avg['CC'][1]:.4f}")
    
    print("\nFrequency Domain Evaluation:")
    print(f"RRMSE: {freq_avg['RRMSE'][0]:.4f} ± {freq_avg['RRMSE'][1]:.4f}")
    print(f"Correlation Coefficient (CC): {freq_avg['CC'][0]:.4f} ± {freq_avg['CC'][1]:.4f}")

    return

# Example usage:
evaluate_model_performance(test_clean, decoded_layer)


#%% post_processing
### 重建的某些EEG信号与ground-truth-EEG在波形上比较类似，但是幅值尺度不匹配，这里后处理，尝试缩放幅值
def scaling(test_clean, decoded_layer):
    decoded_layer_copy = np.copy(decoded_layer)
    for i in range(len(test_clean)):
        scaling_factor = test_clean[i].max() / decoded_layer[i].max()
        decoded_layer_copy[i] = decoded_layer_copy[i]*scaling_factor
    return decoded_layer_copy
    


def linear_scaling(test_clean, decoded_layer):
    decoded_layer_copy = np.copy(decoded_layer)
    for i in range(len(test_clean)):
        data = decoded_layer[i]
        reference = test_clean[i]
        min_signal, max_signal = np.min(data), np.max(data)
        min_ref, max_ref = np.min(reference), np.max(reference)
        scaled_signal = ((data - min_signal) / (max_signal - min_signal)) * (max_ref - min_ref) + min_ref
        decoded_layer_copy[i] = scaled_signal
    return decoded_layer_copy


"""
****
BEST Visualization performance: 
Z-score normalization rescale both 'test_clean' and 'decoded_layer' to have a similar mean and standard deviation 
****
"""
def z_score_normalization(data_array):
    data_copy = np.copy(data_array)
    for i in range(len(data_array)):
        data = data_array[i]
        mean_signal = np.mean(data)
        std_signal = np.std(data)
        normalized_signal = (data - mean_signal) / std_signal
        data_copy[i] = normalized_signal 
    return data_copy




#1
decoded_layer_scaling = scaling(test_clean, decoded_layer)  
#2
decoded_layer_linear_scaling = linear_scaling(test_clean, decoded_layer)
#3 with better visualization performance
test_clean_rescale = z_score_normalization(test_clean)
decoded_layer_rescale = z_score_normalization(decoded_layer)


evaluate_model_performance(test_clean, decoded_layer_scaling)
evaluate_model_performance(test_clean, decoded_layer_linear_scaling)
evaluate_model_performance(test_clean_rescale, decoded_layer_rescale)


    

#%% Impact of the tACS frequencies
def classify_tACS_epochs(eeg_epochs, fs=500):
    """
    Classifies EEG epochs based on dominant frequency into 5Hz, 10Hz, or 40Hz.
    
    Parameters:
        eeg_epochs (np.ndarray): EEG data of shape (n_epochs, n_samples).
        fs (int): Sampling frequency (default: 500Hz).

    Returns:
        dict: Dictionary containing lists of indices for each frequency category.
    """
    eeg_epochs = np.squeeze(eeg_epochs)  # Ensure 2D shape
    freq_bins = {"5Hz": (4, 6), "10Hz": (9, 11), "40Hz": (38, 42)}
    epoch_indices = {key: [] for key in freq_bins}

    for i, epoch in enumerate(eeg_epochs):
        freqs, power = signal.welch(epoch, fs=fs, nperseg=500)
        peak_freq = freqs[np.argmax(power)]

        for label, (f_min, f_max) in freq_bins.items():
            if f_min <= peak_freq <= f_max:
                epoch_indices[label].append(i)
    
    return epoch_indices

def evaluate_model_performance_by_frequency(test_clean, reconstructed_data, classified_epochs, fs=500):
    # Ensure the data is 2D (samples x time points)
    if test_clean.ndim == 3:
        test_clean = test_clean.squeeze()
        reconstructed_data = reconstructed_data.squeeze()

    # Time Domain Evaluation
    # Calculate RRMSE (Relative Root Mean Squared Error)
    def rmsValue(arr):
        square = 0
        mean = 0.0
        root = 0.0
        n = len(arr)
        #Calculate square
        for i in range(0,n):
            square += (arr[i]**2)
        #Calculate Mean
        mean = (square / (float)(n))
        #Calculate Root
        root = math.sqrt(mean)
        return root

    def rrmse(true, pred):
        num = rmsValue(true-pred)
        den = rmsValue(true)
        rrmse_loss = num/den
        return rrmse_loss

    # Calculate Correlation Coefficient (CC) in Time Domain
    def correlation(x, y):
        return pearsonr(x.flatten(), y.flatten())[0]  # Flatten to 1D for CC calculation

    rrmse_time_results = []
    cc_time_results = []

    # Frequency Domain Evaluation (Using Welch Transform)
    # Compute Power Spectral Density (PSD) using Welch's method
    NPERSEG = 200

    rrmse_freq_results = []
    cc_freq_results = []

    for i in range(test_clean.shape[0]):
        # Get the clean, corrupted, and reconstructed data for this sample
        clean_signal = test_clean[i]
        reconstructed_signal = reconstructed_data[i]

        # Calculate Time Domain Metrics
        rrmse_time_results.append(rrmse(clean_signal, reconstructed_signal))
        cc_time_results.append(correlation(clean_signal, reconstructed_signal))

        # Calculate Frequency Domain Metrics
        f, psd_clean = welch(clean_signal, fs=fs, nperseg=NPERSEG)
        _, psd_reconstructed = welch(reconstructed_signal, fs=fs, nperseg=NPERSEG)

        # RRMSE for PSD
        rrmse_freq_results.append(rrmse(psd_clean, psd_reconstructed))

        # Correlation for PSD
        cc_freq_results.append(correlation(psd_clean, psd_reconstructed))

    # Calculate averages and standard deviations for time and frequency domain results
    def calculate_stats(results):
        avg = np.mean(results)
        std_dev = np.std(results)
        return avg, std_dev

    time_avg = {
        'RRMSE': calculate_stats(rrmse_time_results),
        'CC': calculate_stats(cc_time_results)
    }

    freq_avg = {
        'RRMSE': calculate_stats(rrmse_freq_results),
        'CC': calculate_stats(cc_freq_results)
    }

    # Print the results in a readable format (each on a new line)
    print("Time Domain Evaluation:")
    print(f"RRMSE: {time_avg['RRMSE'][0]:.4f} ± {time_avg['RRMSE'][1]:.4f}")
    print(f"Correlation Coefficient (CC): {time_avg['CC'][0]:.4f} ± {time_avg['CC'][1]:.4f}")
    
    print("\nFrequency Domain Evaluation:")
    print(f"RRMSE: {freq_avg['RRMSE'][0]:.4f} ± {freq_avg['RRMSE'][1]:.4f}")
    print(f"Correlation Coefficient (CC): {freq_avg['CC'][0]:.4f} ± {freq_avg['CC'][1]:.4f}")

    # Initialize storage for classified results
    classified_results = {freq: {'rrmse_time': [], 'rrmse_freq': [], 'cc_time': [], 'cc_freq': []} for freq in classified_epochs}
    
    # Loop through each frequency category and store corresponding values
    for freq, indices in classified_epochs.items():
        for i in indices:
            classified_results[freq]['rrmse_time'].append(rrmse_time_results[i])
            classified_results[freq]['rrmse_freq'].append(rrmse_freq_results[i])
            classified_results[freq]['cc_time'].append(cc_time_results[i])
            classified_results[freq]['cc_freq'].append(cc_freq_results[i])
    
    # Function to compute mean and standard deviation
    def calculate_stats(data):
        return np.mean(data), np.std(data)
    
    # Print results
    print("Evaluation Results by tACS Frequency:\n")
    for freq, data in classified_results.items():
        print(f"tACS {freq} Hz ({len(data['rrmse_time'])} epochs classified):")
        
        for key, values in data.items():
            mean, std = calculate_stats(values)
            print(f"  {key}: {mean:.4f} ± {std:.4f}")
        
        print()  # Newline for better readability
        
        
    print('cc_time_max: ', max(cc_time_results))    
    print('cc_freq_max: ', max(cc_freq_results))   
    
    return classified_results


## analysis
classified_epochs = classify_tACS_epochs(test_corrupted)
results_by_freq = evaluate_model_performance_by_frequency(test_clean, decoded_layer, classified_epochs)




#%%
# %% Metadata-based summary table: Overall / Frequency / Amplitude
# This block only generates ONE final result table:
# summary_overall_frequency_amplitude.xlsx

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr


# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

def minmax_normalization_epochwise(data_array):
    """
    Epoch-wise min-max normalization to [0, 1].
    """
    data_array = np.squeeze(data_array)

    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)

    data_norm = np.copy(data_array).astype(np.float64)

    for i in range(data_norm.shape[0]):
        x = data_norm[i]
        x_min = np.min(x)
        x_max = np.max(x)

        if x_max != x_min:
            data_norm[i] = (x - x_min) / (x_max - x_min)
        else:
            data_norm[i] = np.zeros_like(x)

    return data_norm


def rms_value(arr):
    arr = np.asarray(arr)
    return np.sqrt(np.mean(arr ** 2))


def rrmse(true, pred):
    den = rms_value(true)
    if den == 0:
        return np.nan
    return rms_value(true - pred) / den


def cc(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return pearsonr(x, y)[0]


def compute_epoch_metrics(clean_data, reconstructed_data, fs=500):
    """
    Compute epoch-wise RRMSE and CC after epoch-wise [0, 1] min-max normalization.
    """
    clean_data = minmax_normalization_epochwise(clean_data)
    reconstructed_data = minmax_normalization_epochwise(reconstructed_data)

    rows = []

    for i in range(clean_data.shape[0]):
        clean_signal = clean_data[i]
        recon_signal = reconstructed_data[i]

        # Time-domain metrics
        rrmse_t = rrmse(clean_signal, recon_signal)
        cc_t = cc(clean_signal, recon_signal)

        # Frequency-domain metrics
        nperseg = min(200, len(clean_signal))
        _, psd_clean = welch(clean_signal, fs=fs, nperseg=nperseg)
        _, psd_recon = welch(recon_signal, fs=fs, nperseg=nperseg)

        rrmse_f = rrmse(psd_clean, psd_recon)
        cc_f = cc(psd_clean, psd_recon)

        rows.append({
            "RRMSEt": rrmse_t,
            "CCt": cc_t,
            "RRMSEf": rrmse_f,
            "CCf": cc_f
        })

    return pd.DataFrame(rows)


def mean_std_text(values):
    return f"{np.nanmean(values):.4f} ± {np.nanstd(values):.4f}"


def summarize_condition(results_df, condition_name, indices):
    subset = results_df.loc[indices]

    return {
        "Condition": condition_name,
        "N": len(subset),
        "RRMSEt": mean_std_text(subset["RRMSEt"]),
        "CCt": mean_std_text(subset["CCt"]),
        "RRMSEf": mean_std_text(subset["RRMSEf"]),
        "CCf": mean_std_text(subset["CCf"])
    }


# ------------------------------------------------------------
# 2. Check data alignment
# ------------------------------------------------------------

test_clean_2d = np.squeeze(test_clean)
decoded_layer_2d = np.squeeze(decoded_layer)

assert test_clean_2d.shape[0] == decoded_layer_2d.shape[0], \
    "ERROR: test_clean and decoded_layer have different epoch numbers."

assert len(test_metadata) == test_clean_2d.shape[0], \
    "ERROR: test_metadata length does not match test_clean / decoded_layer."

print("Data alignment check passed.")
print("Number of test epochs:", test_clean_2d.shape[0])


# ------------------------------------------------------------
# 3. Compute epoch-wise metrics and combine with metadata
# ------------------------------------------------------------

metric_df = compute_epoch_metrics(
    clean_data=test_clean_2d,
    reconstructed_data=decoded_layer_2d,
    fs=500
)

results_df = pd.concat(
    [test_metadata.reset_index(drop=True), metric_df.reset_index(drop=True)],
    axis=1
)


# ------------------------------------------------------------
# 4. Generate one final summary table
# ------------------------------------------------------------

summary_rows = []

# Overall
summary_rows.append(
    summarize_condition(
        results_df,
        "Overall",
        results_df.index
    )
)

# Frequency-wise
for freq in [5, 10, 40]:
    idx = results_df.index[results_df["frequency_hz"] == freq]
    summary_rows.append(
        summarize_condition(
            results_df,
            f"{freq} Hz",
            idx
        )
    )

# Amplitude-wise
for amp in ["250uA", "1mA"]:
    idx = results_df.index[results_df["amplitude"] == amp]
    summary_rows.append(
        summarize_condition(
            results_df,
            amp,
            idx
        )
    )

summary_table = pd.DataFrame(summary_rows)

# Reorder columns
summary_table = summary_table[
    ["Condition", "N", "RRMSEt", "CCt", "RRMSEf", "CCf"]
]


# ------------------------------------------------------------
# 5. Print and save only one table
# ------------------------------------------------------------

print("\n========== Final Summary Table ==========")
print(summary_table)

output_file = "summary_overall_frequency_amplitude.xlsx"

with pd.ExcelWriter(output_file) as writer:
    summary_table.to_excel(writer, sheet_name="Summary", index=False)

print("\nSaved one final result table:")
print(output_file)


#%% Clean EEG pass-through test
# Purpose:
# Input clean EEG into the trained model and compare model output with the same clean EEG.
# This tests whether the model distorts already-clean EEG epochs.

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr


# ============================================================
# 1. Run clean EEG through the trained model
# ============================================================

clean_input = test_clean.copy()      # shape: [N, samples, 1]
clean_target = test_clean.copy()     # same clean EEG as reference

print("\n========== Clean EEG Pass-through Test ==========")
print("clean_input shape:", clean_input.shape)
print("clean_target shape:", clean_target.shape)

# Use your trained main model
encoded_clean = autoencoder.encoder(clean_input).numpy()
decoded_clean = autoencoder.decoder(encoded_clean).numpy()

clean_output = np.squeeze(decoded_clean)
clean_target_2d = np.squeeze(clean_target)

print("clean_output shape:", clean_output.shape)
print("clean_target_2d shape:", clean_target_2d.shape)

assert clean_output.shape == clean_target_2d.shape, "Shape mismatch between clean output and clean target."


# ============================================================
# 2. Metric functions
# ============================================================

def minmax_epochwise(data):
    """
    Epoch-wise min-max normalization to [0, 1].
    """
    data = np.squeeze(data)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    data_norm = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        x = data[i]
        x_min = np.min(x)
        x_max = np.max(x)

        if x_max > x_min:
            data_norm[i] = (x - x_min) / (x_max - x_min)
        else:
            data_norm[i] = np.zeros_like(x)

    return data_norm


def rms_value(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2))


def rrmse_metric(true, pred):
    denominator = rms_value(true)

    if denominator == 0:
        return np.nan

    return rms_value(true - pred) / denominator


def cc_metric(true, pred):
    true = np.asarray(true).flatten()
    pred = np.asarray(pred).flatten()

    if np.std(true) == 0 or np.std(pred) == 0:
        return np.nan

    return pearsonr(true, pred)[0]


# ============================================================
# 3. Normalize clean target and clean output to [0, 1]
# ============================================================

clean_target_norm = minmax_epochwise(clean_target_2d)
clean_output_norm = minmax_epochwise(clean_output)


# ============================================================
# 4. Compute epoch-wise RRMSE and CC
# ============================================================

rrmse_t_list = []
cc_t_list = []
rrmse_f_list = []
cc_f_list = []

for i in range(clean_target_norm.shape[0]):

    target_epoch = clean_target_norm[i]
    output_epoch = clean_output_norm[i]

    # Time-domain metrics
    rrmse_t = rrmse_metric(target_epoch, output_epoch)
    cc_t = cc_metric(target_epoch, output_epoch)

    rrmse_t_list.append(rrmse_t)
    cc_t_list.append(cc_t)

    # Frequency-domain metrics
    nperseg = min(200, len(target_epoch))

    _, psd_target = welch(target_epoch, fs=sampling_rate, nperseg=nperseg)
    _, psd_output = welch(output_epoch, fs=sampling_rate, nperseg=nperseg)

    rrmse_f = rrmse_metric(psd_target, psd_output)
    cc_f = cc_metric(psd_target, psd_output)

    rrmse_f_list.append(rrmse_f)
    cc_f_list.append(cc_f)


# ============================================================
# 5. Generate one final summary table
# ============================================================

def mean_std_text(values):
    values = np.asarray(values, dtype=np.float64)
    return f"{np.nanmean(values):.4f} ± {np.nanstd(values):.4f}"


clean_pass_summary = pd.DataFrame({
    "Test": ["Clean EEG input → model output"],
    "N": [clean_target_norm.shape[0]],
    "RRMSEt": [mean_std_text(rrmse_t_list)],
    "CCt": [mean_std_text(cc_t_list)],
    "RRMSEf": [mean_std_text(rrmse_f_list)],
    "CCf": [mean_std_text(cc_f_list)]
})


print("\n========== Clean EEG Pass-through Summary ==========")
print(clean_pass_summary)


# ============================================================
# 6. Save only one Excel table
# ============================================================

output_file = "clean_EEG_pass_through_summary.xlsx"

with pd.ExcelWriter(output_file) as writer:
    clean_pass_summary.to_excel(writer, sheet_name="Summary", index=False)

print("\nSaved final table:")
print(output_file)


#%% Plot 8 clean EEG pass-through examples (2x4 subplot)

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Choose which 8 examples to plot
# ============================================================

num_examples = 8

# Use the first 8 epochs by default
example_indices = np.arange(min(num_examples, clean_target_2d.shape[0]))

# If you want random 8 examples instead, use this:
# np.random.seed(1)
# example_indices = np.random.choice(clean_target_2d.shape[0], size=min(num_examples, clean_target_2d.shape[0]), replace=False)

print("Example indices to plot:", example_indices)


# ============================================================
# 2. Optional: normalize to [0,1] for plotting consistency
#    If you want to plot raw waveforms, comment these two lines
#    and use clean_target_2d / clean_output directly below.
# ============================================================

plot_input = minmax_epochwise(clean_target_2d)
plot_output = minmax_epochwise(clean_output)


# ============================================================
# 3. Plot
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for k, idx in enumerate(example_indices):
    ax = axes[k]

    ax.plot(plot_input[idx], label="Input clean EEG", linewidth=1.5)
    ax.plot(plot_output[idx], label="Model output", linewidth=1.5, linestyle='--')

    # If metadata exists, add it to title
    if "test_metadata" in globals():
        meta = test_metadata.iloc[idx]
        title_str = f"Epoch {idx}\n{meta['paradigm']}, {meta['amplitude']}, {meta['frequency_hz']} Hz, Ch{meta['channel']}"
    else:
        title_str = f"Epoch {idx}"

    ax.set_title(title_str, fontsize=10)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

# Hide unused axes if total epochs < 8
for k in range(len(example_indices), len(axes)):
    axes[k].axis("off")

# Add legend only once
axes[0].legend(loc="best", fontsize=9)

plt.suptitle("Clean EEG Pass-through Examples: Input vs Model Output", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("clean_EEG_pass_through_examples.png", dpi=300, bbox_inches='tight')
plt.show()