# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:58:36 2025

Deep Learning for tACS artifacts removal

@author: xingl
"""

#%%
import numpy as np
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

########### readme ###########

# The loaded EEG data is from Alex Casson's lab, which is publicly available so far.
# The loaded EEG data has been preprocessed for further analysis:
# Data Pre-processing Pipeline: 
#     50Hz Notch Filter -> 
#     0.5 Highpass Filter -> 
#     Data Alignment -> 
#     Delete the Approxi. 2-s data in the beginning and end to remove Non-stablized data   

########### readme ###########


#%% data load check
def data_load_check(data_to_check):
    fs=500
    time = np.arange(len(data_to_check[0]))/fs
    plt.figure()
    plt.subplot(311)
    plt.plot(time, data_to_check[0]/1e4)
    plt.subplot(312)
    plt.plot(time, data_to_check[1])
    plt.subplot(313)
    plt.plot(time, data_to_check[2])
    #Perform FFT
    n = len(data_to_check[0])  # Length of the signal
    fft_result = np.fft.fft(data_to_check[0])
    fft_magnitude = np.abs(fft_result) / n  # Normalize the magnitude
    freqs = np.fft.fftfreq(n, d=1/fs)  # Frequency array
    # Take only the positive half of the spectrum
    positive_freqs = freqs[:n // 2]
    positive_magnitude = fft_magnitude[:n // 2]
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_magnitude)
    plt.title("Frequency Spectrum of EEG Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0,60)
    plt.grid()
    plt.show()
    return


## data segment check
def plot_data_pairs(segments, num_pairs=40):
    # Determine the number of rows and columns for subplots (5x8 grid)
    rows = 5
    cols = 8
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()  # Flatten to make it easier to iterate over
    # Loop through the first 'num_pairs' segments
    for i in range(num_pairs):
        clean_segment = segments[i, 0]  # Clean signal
        corrupted_segment = segments[i, 1]  # Corrupted signal
        ax = axes[i]
        ax.plot(clean_segment, label='Clean Signal', color='b')
        ax.plot(corrupted_segment, label='Corrupted Signal', color='r', alpha=0.7)
        ax.set_title(f'Segment {i + 1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        # ax.legend(loc='upper right')
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


#### Note: 电刺激从27.9s开始 (Alpha Data) ####
# Function to segment signals for multiple input signals (2D arrays)
def segment_signals(tacs_start, clean_signal, corrupted_signal, sampling_rate, segment_length_seconds, overlap):
    segment_length_samples = int(segment_length_seconds * sampling_rate)
    step_size = int(segment_length_samples * (1 - overlap))
    start_samples = int(tacs_start * sampling_rate)
    end_samples = start_samples + 58 * sampling_rate  # Adjust the end point for segmentation
    all_segments = []  # List to store the concatenated segments for all signals

    for signal_index in range(clean_signal.shape[0]):
        clean_row = clean_signal[signal_index]
        corrupted_row = corrupted_signal[signal_index]
        segments = []  # Temporary list to store segments for the current signal pair

        for start in range(start_samples, end_samples + 1, step_size):
            end = start + segment_length_samples
            if end <= len(clean_row):  # Check if the segment fits within the signal length
                clean_segment = clean_row[start:end]
                corrupted_segment = corrupted_row[start:end]
                segments.append([clean_segment, corrupted_segment])
            else:
                break

        all_segments.append(np.array(segments))

    # Concatenate segments across all rows (signals)
    all_segments = np.concatenate(all_segments, axis=0)
    return all_segments


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


#%%
data_load_check(EEG_ground_truth_alpha)


#%% Data Segmentation

tacs_start_alpha = 27.9+1
# Parameters for segmentation (alpha EEG data)
tacs_start_erp = 26.5+1  # Start of the tACS erp data
sampling_rate = 500  # Sampling rate
segment_length_seconds = 2  # Length of each segment in seconds
overlap = 0.5  # Overlap between segments

# alpha data
segments1 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_1mA_5Hz_alpha, sampling_rate, segment_length_seconds, overlap)
segments2 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_1mA_10Hz_alpha, sampling_rate, segment_length_seconds, overlap)
segments3 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_1mA_40Hz_alpha, sampling_rate, segment_length_seconds, overlap)
segments4 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_250uA_5Hz_alpha, sampling_rate, segment_length_seconds, overlap)
segments5 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_250uA_10Hz_alpha, sampling_rate, segment_length_seconds, overlap)
segments6 = segment_signals(tacs_start_alpha, EEG_ground_truth_alpha, Test_250uA_40Hz_alpha, sampling_rate, segment_length_seconds, overlap)

## erp data
segments11 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_1mA_5Hz_erp, sampling_rate, segment_length_seconds, overlap)
segments22 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_1mA_10Hz_erp, sampling_rate, segment_length_seconds, overlap)
segments33 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_1mA_40Hz_erp, sampling_rate, segment_length_seconds, overlap)
segments44 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_250uA_5Hz_erp, sampling_rate, segment_length_seconds, overlap)
segments55 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_250uA_10Hz_erp, sampling_rate, segment_length_seconds, overlap)
segments66 = segment_signals(tacs_start_erp, EEG_ground_truth_erp, Test_250uA_40Hz_erp, sampling_rate, segment_length_seconds, overlap)


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
    
    

#%% 
plot_data_pairs(segments1, num_pairs=40)
plot_data_pairs(normalized_segments1, num_pairs=40)
plot_data_pairs(all_segments, num_pairs=40)



#%% Deep learning model 
######  Define a convolutional Autoencoder
import tensorflow as tf
from tensorflow.keras import layers, Model, losses
from tensorflow import keras

class AutoencoderX(Model):
    def __init__(self):
        super(AutoencoderX, self).__init__()
        self.encoder = keras.Sequential([
            layers.Input(shape=(1000, 1)),
            layers.Conv1D(64, 3, activation=None, padding='same', strides=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1D(32, 3, activation=None, padding='same', strides=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1D(16, 3, activation=None, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1D(8, 3, activation=None, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1D(4, 3, activation=None, padding='same', strides=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.2)
        ])

        self.decoder = keras.Sequential([
            layers.Conv1D(8, 3, activation=None, padding='same', strides=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1DTranspose(16, 3, activation=None, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1DTranspose(32, 3, activation=None, padding='same', strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1D(64, 3, activation=None, padding='same', strides=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded





#%% Cross_validation
# Define function to perform KFold cross-validation
def kfold_cross_validation(corrupted_data, clean_data, num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_losses = []  # To store the losses for each fold
    fold_accuracies = []  # To store the accuracies for each fold
    best_loss = float('inf')  # Initialize the best loss to infinity
    best_model = None  # To store the best model

    for fold, (train_indices, test_indices) in enumerate(kf.split(corrupted_data)):
        print(f"\nTraining fold {fold + 1}...")
        
        # Split data into training and test sets for the current fold
        train_corrupted = corrupted_data[train_indices]
        train_clean = clean_data[train_indices]
        test_corrupted = corrupted_data[test_indices]
        test_clean = clean_data[test_indices]

        # Reshape data to match model input shape
        train_corrupted = np.expand_dims(train_corrupted, -1)
        train_clean = np.expand_dims(train_clean, -1)
        test_corrupted = np.expand_dims(test_corrupted, -1)
        test_clean = np.expand_dims(test_clean, -1)
        
        # Build and compile the autoencoder model
        autoencoder = AutoencoderX()
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
        
        # Train the model
        autoencoder.fit(train_corrupted, train_clean, epochs=1000, batch_size=32, verbose=1)
        
        # Evaluate the model on the test set
        test_loss, test_accuracy = autoencoder.evaluate(test_corrupted, test_clean, verbose=1)
        print(f"Test Loss (MSE) for fold {fold + 1}: {test_loss}")
        print(f"Test Accuracy for fold {fold + 1}: {test_accuracy}")
        
        # Append the results for this fold
        fold_losses.append(test_loss)
        fold_accuracies.append(test_accuracy)
        
        # Save the best model (based on test loss)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = autoencoder
            best_model.save("10foldCV/best_AutoencoderX.keras")  
            best_model.save("10foldCV/best_AutoencoderX.h5")

    # After all folds, print the average and standard deviation of loss and accuracy
    print("\nKFold Cross-Validation Results:")
    print(f"Average Test Loss (MSE): {np.mean(fold_losses)} ± {np.std(fold_losses)}")
    print(f"Average Test Accuracy: {np.mean(fold_accuracies)} ± {np.std(fold_accuracies)}")
    
    return best_model


# Assuming all_segments is of shape [2178, 2, 1000]
# all_segments[:, 0, :] will be the clean data
# all_segments[:, 1, :] will be the corrupted data

# Split the data into corrupted and clean datasets
corrupted_data = all_segments[:, 1, :]  # will be the clean data
clean_data = all_segments[:, 0, :]  # will be the corrupted data

# Perform KFold Cross-Validation and get the best model
best_model = kfold_cross_validation(corrupted_data, clean_data, num_folds=10)


#%% Build model and test
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses

train_data, tem_data = train_test_split(all_segments, test_size=0.2, random_state=1, shuffle=True)
valid_data, test_data = train_test_split(tem_data, test_size=0.5, random_state=1, shuffle=True)

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
    filepath="best_autoencoder",  # No .h5 extension → TensorFlow SavedModel format
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
    save_format="tf"  # Save using TensorFlow format instead of HDF5
)


# Compile the autoencoder model
autoencoder = AutoencoderX()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

# Train the model
Epoch = 1000
history = autoencoder.fit(
    train_corrupted, train_clean, 
    epochs=Epoch, 
    shuffle=True, 
    validation_data=(valid_corrupted, valid_clean),
    callbacks=[checkpoint_callback]
)

# Load the best model
best_autoencoder = tf.keras.models.load_model("best_autoencoder")

# Model details
best_autoencoder.encoder.summary()
best_autoencoder.decoder.summary()


encoded_layer = best_autoencoder.encoder(test_corrupted).numpy()
decoded_layer = best_autoencoder.decoder(encoded_layer).numpy()
decoded_layer = np.squeeze(decoded_layer) # back to 2-dimensional array



## save model
# autoencoder.save("saved_model/autoencoderX.h5")  
# autoencoder.save('saved_model/AutoencoderX.keras')
# # Convert the model to TensorFlow Lite and save
# converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
# tflite_model = converter.convert()
# with open('saved_model/AutoencoderX.tflite', 'wb') as f:
#   f.write(tflite_model)





#%%
fs = 500
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


    
#%% plot
def plot3versions(test_corrupted, test_clean, decoded_layer):
    import matplotlib.gridspec as gridspec
    # gridspec inside gridspec
    fig = plt.figure(figsize=(20,20))
    gs0 = gridspec.GridSpec(2, 4, figure=fig)
    fs=500
    time = np.linspace(0, len(test_clean[0])/fs, num=len(test_clean[0]))
    
    idx = 100
    
    for i in range(8):
        gs00 = gs0[i].subgridspec(3, 1)
        ax1 = fig.add_subplot(gs00[0:1,:])
        plt.plot(time, test_corrupted[i+idx], label="Contaminated EEG",color='tab:blue')
        ax1.get_xaxis().set_visible(False)
        plt.ylabel(r'Normalized amplitude')
        plt.rcParams.update({'font.size': 16})
        
        if i==0:
            plt.legend()
        
        ax2 = fig.add_subplot(gs00[1:3,:])
        plt.plot(time, test_clean[i+idx], label="Ground-truth clean EEG",color='tab:green')
        plt.plot(time, decoded_layer[i+idx], label="Reconstructed EEG",linestyle='dashed',linewidth=2,color = 'tab:orange')
        plt.xlabel('Time (s)')
        plt.ylabel(r'Rescaled amplitude')
        plt.rcParams.update({'font.size': 16})
        
        if i==0:
            plt.legend()
        
    plt.tight_layout()
    # plt.savefig('AutoencoderX_tACS_removal_09.pdf')
    plt.show()
    return


plot3versions(test_corrupted, test_clean_rescale, decoded_layer_rescale)


#%% Amplitude recovery via Z-score
"""
原来的EEG amplitude是采集的模拟信号，幅值也不正常
"""
def recover_amplitude(normalized_epochs, reference_signal):
    """Rescale using Z-score"""
    ref_mean = np.mean(reference_signal)
    ref_std = np.std(reference_signal)
    normalized_epochs_copy = normalized_epochs
    recovered_epochs = normalized_epochs_copy * ref_std + ref_mean
    return recovered_epochs

def rescale_epoch(normalized_epoch, reference_segment):
    """ Rescales a normalized EEG epoch [0,1] using the reference EEG segment. """
    min_ref, max_ref = np.min(reference_segment), np.max(reference_segment)
    return normalized_epoch * (max_ref - min_ref) + min_ref



## 取一整段测试信号
ground_truth_alpha = EEG_ground_truth_alpha[0,:]
test_eeg_signal = Test_1mA_10Hz_alpha[0,:]
fs=500
time = np.linspace(0, len(test_eeg_signal)/fs, num=len(test_eeg_signal))
# plt.figure()
# plt.plot(time,test_eeg_signal)
# plt.plot([tacs_start, tacs_start], [-1000000, 1000000])

## 取tACS开始之前的干净信号，缩放到正常的EEG幅值范围
tacs_start = 27.9
scaling_factor = 5000
eeg_before_tacs = test_eeg_signal[0:int(tacs_start*fs)] / scaling_factor
# plt.figure()
# plt.plot(eeg_before_tacs)


eeg_during_tacs = test_eeg_signal[int(tacs_start*fs): int((tacs_start+60)*fs)]



epoch_length= 2*fs
num_epochs = int(len(eeg_during_tacs)/epoch_length)



# Process EEG signal
processed_outputs = []

for i in range(num_epochs):
    # Step 1: Segment the EEG signal
    epoch = eeg_during_tacs[i * epoch_length:(i + 1) * epoch_length]
    
    # Step 2: Min-Max Normalize the Epoch
    min_val, max_val = np.min(epoch), np.max(epoch)
    normalized_epoch = (epoch - min_val) / (max_val - min_val)
    normalized_epoch = normalized_epoch.reshape(1,1000,1)
    
    # Step 3: Input to DL Model
    encoded = best_autoencoder.encoder(normalized_epoch).numpy()
    output = best_autoencoder.decoder(encoded).numpy()
    output = np.squeeze(output) # back to 2-dimensional array

    # Step 4: Z-Score Denormalization (based on the clean reference signal)
    # denormalized_output = recover_amplitude(output, eeg_before_tacs)
    denormalized_output = rescale_epoch(output, eeg_before_tacs)

    # Store processed output
    processed_outputs.append(denormalized_output)


# Convert to a single array
final_denormalized_signal = np.concatenate(processed_outputs)
final_eeg_signal = np.concatenate((eeg_before_tacs, final_denormalized_signal)) 


plt.figure()
plt.plot(final_eeg_signal)


ground_truth_alpha_crop = ground_truth_alpha[int(tacs_start*fs): int((tacs_start+60)*fs)] / scaling_factor
ground_truth_alpha_crop = ground_truth_alpha_crop - np.mean(ground_truth_alpha_crop)

x111 = rrmse(ground_truth_alpha_crop, final_denormalized_signal)
x222 = correlation(ground_truth_alpha_crop, final_denormalized_signal)
print('time rrmse: ', x111)
print('time cc: ', x222)

plt.figure()
plt.plot(ground_truth_alpha_crop)
plt.plot(final_denormalized_signal)


#%% Plot Learning Curve
plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1) 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.title('Learning Curve - Loss') 
plt.legend() 

plt.subplot(1, 2, 2) 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.title('Learning Curve - Accuracy') 
plt.legend() 
plt.tight_layout() 
# plt.savefig('learning_curve_autoencoderX.pdf')
plt.show()



#%% plot preprocessed data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker  # Import ticker for formatting

# Use seaborn style with background grid
sns.set(style="darkgrid")

fs = 500  # Sampling rate
times = np.linspace(0, len(EEG_ground_truth_erp[0, :]) / fs, num=len(EEG_ground_truth_erp[0, :]))

# Define epoch range for highlighting
epoch_start, epoch_end = 26.5, 86.5

# Create figure
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# EEG Data and labels
eeg_data = [
    (EEG_ground_truth_erp[0, :], "Ground-truth EEG"),
    (Test_1mA_5Hz_erp[0, :], "EEG-tACS-5Hz-1mA"),
    (Test_1mA_10Hz_erp[0, :], "EEG-tACS-10Hz-1mA"),
    (Test_1mA_40Hz_erp[0, :], "EEG-tACS-40Hz-1mA")
]

# Define common font sizes
label_fontsize = 18
tick_fontsize = 16
legend_fontsize = 18

# Set up y-axis formatting
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))  # Forces scientific notation for values outside this range

# Plot each EEG signal
for ax, (data, label) in zip(axes, eeg_data):
    ax.plot(times, data, label=label, color="b", linewidth=1)  
    ax.axvspan(epoch_start, epoch_end, color='red', alpha=0.2)  # Add transparent red background

    # Customize grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    
    # Set labels and legend
    ax.set_ylabel("Amplitude", fontsize=label_fontsize)
    ax.legend(loc="upper left", fontsize=legend_fontsize, frameon=True)  # Adjust legend position
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Apply consistent y-axis formatting
    ax.yaxis.set_major_formatter(formatter) 

# Final adjustments
axes[-1].set_xlabel("Time (s)", fontsize=label_fontsize)
plt.tight_layout() 
# plt.savefig('preprocessed_data.pdf')
plt.show()



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


#%% plot3versionsEnvelope
def plot3versionsEnvelope(test_corrupted, test_clean, decoded_layer):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from scipy.signal import find_peaks
    
    # gridspec inside gridspec
    fig = plt.figure(figsize=(20,10))
    gs0 = gridspec.GridSpec(2, 4, figure=fig)
    fs = 500
    time = np.linspace(0, len(test_clean[0])/fs, num=len(test_clean[0]))
    
    idx = 50
    
    for i in range(8):
        gs00 = gs0[i].subgridspec(3, 1)
        ax1 = fig.add_subplot(gs00[0:1, :])
        signal_1d = test_corrupted[i+idx].flatten()
        peaks, _ = find_peaks(signal_1d)
        troughs, _ = find_peaks(-signal_1d)
        
        plt.plot(time, signal_1d, label="Contaminated EEG", color='tab:blue')
        plt.plot(time[peaks], signal_1d[peaks], marker='o', markersize=3, linestyle='-', color='tab:red', label="Peak Envelope")
        plt.plot(time[troughs], signal_1d[troughs], marker='o', markersize=3, linestyle='-', color='tab:red')
        
        ax1.get_xaxis().set_visible(False)
        plt.ylabel(r'Amplitude ($\mu$V)')
        plt.rcParams.update({'font.size': 16})
        
        if i == 0:
            plt.legend()
        
        ax2 = fig.add_subplot(gs00[1:3, :])
        plt.plot(time, test_clean[i+idx], label="Ground-truth clean EEG", color='tab:green')
        plt.plot(time, decoded_layer[i+idx], label="Reconstructed EEG", linestyle='dashed', linewidth=2, color='tab:orange')
        plt.xlabel('Time (s)')
        plt.ylabel(r'Normalized amplitude')
        plt.rcParams.update({'font.size': 16})
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('3plots_tACS_envelope.pdf')
    plt.show()
    return

plot3versionsEnvelope(test_corrupted, test_clean_rescale, decoded_layer_rescale)








