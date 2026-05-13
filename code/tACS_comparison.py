# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:05:24 2025

@author: xingl
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim

from scipy.signal import welch
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses

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



### Data Segmentation
# Parameters for segmentation (alpha EEG data)

tacs_start_alpha =27.9 +1# alpha data
tacs_start_erp = 26.5 +1 # erp data
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


### Data Normalization
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
# Load the best model
best_autoencoder = tf.keras.models.load_model("best_autoencoder")

# Model details
best_autoencoder.encoder.summary()
best_autoencoder.decoder.summary()


encoded_layer = best_autoencoder.encoder(test_corrupted).numpy()
decoded_layer = best_autoencoder.decoder(encoded_layer).numpy()
# decoded_layer = np.squeeze(decoded_layer) # back to 2-dimensional array




#%% EEG before and after tACS removal ---- time-frequency comparison

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.stats import pearsonr

# ==========================================================
# Settings
# ==========================================================
fs = 500                      # sampling rate, Hz
wavelet = "morl"              # Morlet wavelet
freq_min = 1                  # Hz
freq_max = 45                 # Hz
num_freqs = 100
sample_id = 5                # choose which test segment to visualize

# ==========================================================
# Prepare data shape: [num_samples, 1000]
# ==========================================================
x_corrupted = np.squeeze(test_corrupted)
x_clean = np.squeeze(test_clean)
x_reconstructed = np.squeeze(decoded_layer)

print("Corrupted shape:", x_corrupted.shape)
print("Clean shape:", x_clean.shape)
print("Reconstructed shape:", x_reconstructed.shape)

num_samples, signal_len = x_clean.shape
time_axis = np.arange(signal_len) / fs

# ==========================================================
# CWT function
# ==========================================================
def compute_cwt(signal_1d, fs, freq_min=1, freq_max=80, num_freqs=100, wavelet="morl"):
    freqs = np.linspace(freq_min, freq_max, num_freqs)
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq * fs / freqs

    coeffs, _ = pywt.cwt(signal_1d, scales, wavelet, sampling_period=1/fs)
    power = np.abs(coeffs)

    return power, freqs

# ==========================================================
# Compute CWT similarity statistics
# Clean CWT is used as the ground truth
# ==========================================================
corr_clean_recon = []
corr_clean_corrupted = []

for i in range(num_samples):

    cwt_clean, freqs = compute_cwt(
        x_clean[i], fs, freq_min, freq_max, num_freqs, wavelet
    )

    cwt_recon, _ = compute_cwt(
        x_reconstructed[i], fs, freq_min, freq_max, num_freqs, wavelet
    )

    cwt_corrupted, _ = compute_cwt(
        x_corrupted[i], fs, freq_min, freq_max, num_freqs, wavelet
    )

    # Flatten 2D scalograms and compute Pearson correlation
    r_recon, _ = pearsonr(cwt_clean.flatten(), cwt_recon.flatten())
    r_corrupted, _ = pearsonr(cwt_clean.flatten(), cwt_corrupted.flatten())

    corr_clean_recon.append(r_recon)
    corr_clean_corrupted.append(r_corrupted)

corr_clean_recon = np.array(corr_clean_recon)
corr_clean_corrupted = np.array(corr_clean_corrupted)

print("\n===== CWT Scalogram Similarity Statistics =====")
print("Clean vs Reconstructed:")
print(f"Mean correlation = {np.mean(corr_clean_recon):.4f}")
print(f"Std correlation  = {np.std(corr_clean_recon):.4f}")

print("\nClean vs Corrupted:")
print(f"Mean correlation = {np.mean(corr_clean_corrupted):.4f}")
print(f"Std correlation  = {np.std(corr_clean_corrupted):.4f}")

# ==========================================================
# Helper: min-max normalization to [0, 1]
# ==========================================================
def minmax_norm(x):
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)



# ==========================================================
# Plot one example: also print CC of current plotted case
# ==========================================================

sig_corrupted = x_corrupted[sample_id]

sig_clean = minmax_norm(x_clean[sample_id])
sig_recon = minmax_norm(x_reconstructed[sample_id])

# CWT after normalization
cwt_corrupted, freqs = compute_cwt(sig_corrupted, fs, freq_min, freq_max, num_freqs, wavelet)
cwt_clean, _ = compute_cwt(sig_clean, fs, freq_min, freq_max, num_freqs, wavelet)
cwt_recon, _ = compute_cwt(sig_recon, fs, freq_min, freq_max, num_freqs, wavelet)

# ==========================================================
# Correlation coefficients for current plotted sample
# ==========================================================
cc_time, _ = pearsonr(sig_clean.flatten(), sig_recon.flatten())
cc_cwt, _ = pearsonr(cwt_clean.flatten(), cwt_recon.flatten())

print("====================================")
print("Current plotted sample ID:", sample_id)
print("Time-domain CC (GT vs Recon): %.4f" % cc_time)
print("CWT-domain  CC (GT vs Recon): %.4f" % cc_cwt)
print("====================================")

# ==========================================================
# Better plotting style for paper
# ==========================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11
})

fig, axes = plt.subplots(2, 3, figsize=(24, 8), constrained_layout=True)

# --------------------------
# Time-domain plots
# --------------------------
axes[0,0].plot(time_axis, sig_corrupted, color="tab:blue", linewidth=1.8)
axes[0,0].set_title("Corrupted EEG")
axes[0,0].set_xlabel("Time (s)")
axes[0,0].set_ylabel("Amplitude")

axes[0,1].plot(time_axis, sig_clean, color="tab:green", linewidth=1.8)
axes[0,1].set_title("Ground-truth EEG")
axes[0,1].set_xlabel("Time (s)")
axes[0,1].set_ylabel("Normalized")

axes[0,2].plot(time_axis, sig_clean, color="tab:green",
               linewidth=1.8, label="Ground-truth")

axes[0,2].plot(time_axis, sig_recon,
               color="tab:orange",
               linestyle="--",
               linewidth=2.0,
               label="Reconstructed")

axes[0,2].set_title(f"Recon EEG (Time domain CC={cc_time:.3f})")
axes[0,2].set_xlabel("Time (s)")
axes[0,2].set_ylabel("Normalized")
axes[0,2].legend(frameon=False)

# --------------------------
# CWT plots
# --------------------------
extent = [time_axis[0], time_axis[-1], freqs[0], freqs[-1]]

im0 = axes[1,0].imshow(
    cwt_corrupted,
    extent=extent,
    aspect="auto",
    origin="lower",
    cmap="turbo"
)
axes[1,0].set_title("CWT: Corrupted EEG")
axes[1,0].set_xlabel("Time (s)")
axes[1,0].set_ylabel("Frequency (Hz)")
axes[1,0].set_ylim([0,45])
fig.colorbar(im0, ax=axes[1,0], shrink=0.9)

im1 = axes[1,1].imshow(
    cwt_clean,
    extent=extent,
    aspect="auto",
    origin="lower",
    cmap="turbo"
)
axes[1,1].set_title("CWT: Ground-truth EEG")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("Frequency (Hz)")
axes[1,1].set_ylim([0,45])
fig.colorbar(im1, ax=axes[1,1], shrink=0.9)

im2 = axes[1,2].imshow(
    cwt_recon,
    extent=extent,
    aspect="auto",
    origin="lower",
    cmap="turbo"
)
axes[1,2].set_title(f"CWT: Recon (Scalogram CC={cc_cwt:.3f})")
axes[1,2].set_xlabel("Time (s)")
axes[1,2].set_ylabel("Frequency (Hz)")
axes[1,2].set_ylim([0,45])
fig.colorbar(im2, ax=axes[1,2], shrink=0.9)



# ==========================================================
# Save as PDF
# ==========================================================
plt.savefig(
    "EEG_CWT_Comparison.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()



#%% model inference time
# ==========================================================
# TensorFlow Online Inference Latency Benchmark (CPU / GPU)
# Using actual test data: test_corrupted
# Shape = (213, 1000, 1)
# ==========================================================

import os
import time
import numpy as np
import tensorflow as tf

# ==========================================================
# USER SETTINGS
# ==========================================================
MODEL_PATH = "best_autoencoder"
USE_DEVICE = "CPU"        # change to "CPU" or "GPU"

# ==========================================================
# SELECT DEVICE
# ==========================================================
if USE_DEVICE.upper() == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("TensorFlow Version:", tf.__version__)
print("Available Devices:", tf.config.list_physical_devices())

# ==========================================================
# LOAD MODEL
# ==========================================================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================================
# FAST INFERENCE GRAPH
# ==========================================================
@tf.function
def run_model(x):
    return model(x, training=False)

# ==========================================================
# CHECK INPUT DATA
# test_corrupted should already exist in memory
# shape = (213,1000,1)
# ==========================================================
print("Input data shape:", test_corrupted.shape)

num_samples = test_corrupted.shape[0]

# ==========================================================
# WARM-UP
# ==========================================================
print("Warming up...")

dummy = test_corrupted[0:1].astype(np.float32)

for _ in range(10):
    _ = run_model(dummy)

# ==========================================================
# ONLINE TEST
# One sample arrives at a time
# ==========================================================
latencies = []

print("Running benchmark...")

for i in range(num_samples):

    # get one incoming sample
    x = test_corrupted[i:i+1].astype(np.float32)   # shape (1,1000,1)

    t0 = time.perf_counter()

    output = run_model(x)
    output = output.numpy()

    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    latencies.append(latency_ms)

# ==========================================================
# RESULTS
# ==========================================================
latencies = np.array(latencies)

print("\n==============================")
print("DEVICE:", USE_DEVICE)
print("Samples Tested:", num_samples)
print("==============================")
print("Average Latency : %.4f ms" % np.mean(latencies))
print("Std Dev         : %.4f ms" % np.std(latencies))
print("Min Latency     : %.4f ms" % np.min(latencies))
print("Max Latency     : %.4f ms" % np.max(latencies))
print("Throughput      : %.2f windows/sec" % (1000 / np.mean(latencies)))
print("==============================")



















#%% SMA and AF test for long signal test
# from sma_artifact_removal import SMAMethod

# tacs_freq = 5
# fs= 500
# contaminated = Test_1mA_5Hz_alpha[0, int(tacs_start_alpha*fs):int((tacs_start_alpha+60)*fs)]
# time = np.arange(len(contaminated))/fs
# sma = SMAMethod(tacs_freq=tacs_freq, sampling_rate=fs)
# cleaned = sma.apply(contaminated)

# plt.figure(figsize=(12,6))
# plt.plot(time, contaminated, label='Contaminated EEG')
# plt.plot(time, cleaned, label='Cleaned EEG (SMA)', alpha=0.7)
# plt.legend()
# plt.xlabel('Time (s)')
# plt.title('SMA Artifact Removal Example')
# plt.show()



# from af_artifact_removal import AFMethod

# af = AFMethod(forgetting_factor=0.999)
# cleaned_signal = af.apply(contaminated, tacs_freq=5, sampling_rate=500)

# plt.figure(figsize=(12,6))
# plt.plot(time, contaminated, label='Contaminated EEG')
# plt.plot(time, cleaned, label='Cleaned EEG (AF)', alpha=0.7)
# plt.legend()
# plt.xlabel('Time (s)')
# plt.title('Adaptive Filtering Artifact Removal Example')
# plt.show()


# #%% apply SMA and AF to test_corrupted data
# corrupted_data_test = np.squeeze(test_corrupted[0])
# time = np.arange(len(corrupted_data_test))/fs
# sma = SMAMethod(tacs_freq = 10, sampling_rate=fs)
# sma_cleaned_test = sma.apply(corrupted_data_test)
# af = AFMethod(forgetting_factor=0.999)
# af_cleaned_test = af.apply(corrupted_data_test, tacs_freq=10, sampling_rate=500)
# #%%
# plt.figure(figsize=(12, 12))
# plt.subplot(511)
# plt.plot(time, corrupted_data_test, label="Corrupted EEG", color='red')
# plt.legend()
# plt.subplot(512)
# plt.plot(time, np.squeeze(test_clean[0]), label="Ground-truth EEG", color='blue')
# plt.legend()
# plt.subplot(513)
# plt.plot(time, decoded_layer[0], label="DL cleaned EEG", color='green')
# plt.legend()
# plt.subplot(514)
# plt.plot(time, sma_cleaned_test, label="SMA cleaned EEG", color='orange')
# plt.legend()
# plt.subplot(515)
# plt.plot(time, af_cleaned_test, label="AF cleaned EEG", color='purple')
# plt.legend()
# plt.tight_layout()
# plt.show()

#%%

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
    detected_frequencies = []

    for i, epoch in enumerate(eeg_epochs):
        freqs, power = signal.welch(epoch, fs=fs, nperseg=500)
        peak_freq = freqs[np.argmax(power)]
        detected_frequencies.append(peak_freq)

        for label, (f_min, f_max) in freq_bins.items():
            if f_min <= peak_freq <= f_max:
                epoch_indices[label].append(i)
    
    return epoch_indices, detected_frequencies



classified_epochs, detected_frequencies = classify_tACS_epochs(test_corrupted)
for key, value in classified_epochs.items():
    print(f"{key}: {len(value)}")

#%% comparison
import time
from sma_artifact_removal import SMAMethod
from af_artifact_removal import AFMethod
DL_cleaned=[]
SMA_cleaned=[]
AF_cleaned=[]
Detected_freq = []
sma_times = []
af_times = []
fs=500
for i in range(np.shape(test_corrupted)[0]):
    # DL
    corrupted_eeg_epoch = np.squeeze(test_corrupted[i])
    ground_truth_eeg = np.squeeze(test_clean[i])
    DL_cleaned_eeg = decoded_layer[i]
    
    # Frequency detection
    detected_tacs_freq = detected_frequencies[i]

    # SMA timing
    start_sma = time.time()
    sma = SMAMethod(tacs_freq=detected_tacs_freq, sampling_rate=fs)
    sma_cleaned_eeg = sma.apply(corrupted_eeg_epoch)
    end_sma = time.time()
    sma_times.append(end_sma - start_sma)

    # AF timing
    start_af = time.time()
    af = AFMethod(forgetting_factor=0.999)
    af_cleaned_eeg = af.apply(corrupted_eeg_epoch, tacs_freq=detected_tacs_freq, sampling_rate=500)
    end_af = time.time()
    af_times.append(end_af - start_af)

    # Append results
    DL_cleaned.append(DL_cleaned_eeg)
    SMA_cleaned.append(sma_cleaned_eeg)
    AF_cleaned.append(af_cleaned_eeg)

# Compute and print average execution times
avg_sma_time = np.mean(sma_times)
avg_af_time = np.mean(af_times)

print(f"Average SMA execution time: {avg_sma_time:.6f} seconds")
print(f"Average AF execution time: {avg_af_time:.6f} seconds")

    
    
# convert to array
DL_cleaned_array = np.array(DL_cleaned)
SMA_cleaned_array = np.array(SMA_cleaned)
AF_cleaned_array = np.array(AF_cleaned)


def z_score_normalization(data_array):
    data_copy = np.copy(data_array)
    for i in range(len(data_array)):
        data = data_array[i]
        mean_signal = np.mean(data)
        std_signal = np.std(data)
        normalized_signal = (data - mean_signal) / std_signal
        data_copy[i] = normalized_signal 
    return data_copy


def min_max_normalization(data_array):
    data_copy = np.copy(data_array)
    for i in range(len(data_array)):
        data = data_array[i]
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_signal = (data - min_val) / (max_val - min_val)
        data_copy[i] = normalized_signal
    return data_copy


# Rescale: with better visualization performance
test_clean_rescale = test_clean
DL_cleaned_array_rescale = DL_cleaned_array
SMA_cleaned_array_rescale = min_max_normalization(SMA_cleaned_array)
AF_cleaned_array_rescale = min_max_normalization(AF_cleaned_array)



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

print('\n DL:  \n')
evaluate_model_performance(test_clean_rescale, DL_cleaned_array_rescale)

print('\n SMA: \n')
evaluate_model_performance(test_clean_rescale, SMA_cleaned_array_rescale)

print('\n AF: \n')
evaluate_model_performance(test_clean_rescale, AF_cleaned_array_rescale)    


# evaluate_model_performance_by_frequency(test_clean, DL_cleaned_array, classified_epochs)
# evaluate_model_performance_by_frequency(test_clean, SMA_cleaned_array, classified_epochs)
# evaluate_model_performance_by_frequency(test_clean, AF_cleaned_array, classified_epochs)    




#%% plot 3 comparisons
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot3versions(test_corrupted, test_clean, DL_cleaned_array, SMA_cleaned_array, AF_cleaned_array):
    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
    fig.patch.set_facecolor('white')
    fs = 500
    idx = 51
    time = np.linspace(0, len(test_clean[0]) / fs, num=len(test_clean[0]))
    test_clean = np.squeeze(test_clean)

    # Subplot 1: Contaminated EEG
    axs[0].plot(time, test_corrupted[idx], label="Contaminated EEG", color='tab:blue')
    axs[0].set_ylabel('Normalized\namplitude')
    axs[0].legend(loc='upper right')
    axs[0].set_title("EEG Artifact Removal Comparison", fontsize=20)

    # Subplot 2: Ground-truth vs tACSNet
    axs[1].plot(time, test_clean[idx], label="Ground-truth clean EEG", color='tab:green')
    axs[1].plot(time, DL_cleaned_array[idx], label="tACSNet cleaned EEG", linestyle='dashed', linewidth=2, color='darkorange')
    cc_dl = np.corrcoef(test_clean[idx], DL_cleaned_array[idx])[0, 1]
    axs[1].text(0.01, 0.9, f'$CC_{{t}}$ = {cc_dl:.3f}', transform=axs[1].transAxes,
                fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))
    axs[1].set_ylabel('Rescaled\namplitude')
    axs[1].legend(loc='upper right')

    # Subplot 3: Ground-truth vs SMA
    axs[2].plot(time, test_clean[idx], label="Ground-truth clean EEG", color='tab:green')
    axs[2].plot(time, SMA_cleaned_array[idx], label="SMA cleaned EEG", linestyle='dashed', linewidth=2, color='purple')
    cc_sma = np.corrcoef(test_clean[idx], SMA_cleaned_array[idx])[0, 1]
    axs[2].text(0.01, 0.9, f'$CC_{{t}}$ = {cc_sma:.3f}', transform=axs[2].transAxes,
                fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))
    axs[2].set_ylabel('Rescaled\namplitude')
    axs[2].legend(loc='upper right')

    # Subplot 4: Ground-truth vs AF
    axs[3].plot(time, test_clean[idx], label="Ground-truth clean EEG", color='tab:green')
    axs[3].plot(time, AF_cleaned_array[idx], label="AF cleaned EEG", linestyle='dashed', linewidth=2, color='magenta')
    cc_af = np.corrcoef(test_clean[idx], AF_cleaned_array[idx])[0, 1]
    axs[3].text(0.01, 0.9, f'$CC_{{t}}$ = {cc_af:.3f}', transform=axs[3].transAxes,
                fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))
    axs[3].set_ylabel('Rescaled\namplitude')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')

    for ax in axs:
        ax.grid(False)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    plt.tight_layout()
    # plt.savefig('Algorithm_comparison_best_1.pdf', dpi=300, facecolor='white')
    plt.show()






plot3versions(test_corrupted, test_clean_rescale, DL_cleaned_array_rescale, SMA_cleaned_array_rescale, AF_cleaned_array_rescale)



#%%


import numpy as np
import matplotlib.pyplot as plt

# Algorithms and metrics
algorithms = ['tACSNet', 'SMA', 'AF']
metrics = ['RRMSE_t', 'RRMSE_f', 'CC_t', 'CC_f']

# Mean values
means = {
    'tACSNet': [0.2832, 0.6463, 0.6387, 0.8801],
    'SMA':     [0.3387, 0.7212, 0.6225, 0.8004],
    'AF':      [0.3481, 1.3038, 0.5890, 0.6695]
}

# Standard deviations
stds = {
    'tACSNet': [0.0874, 0.2359, 0.2030, 0.1147],
    'SMA':     [0.0939, 0.3652, 0.1272, 0.1727],
    'AF':      [0.0726, 0.8678, 0.1591, 0.2381]
}


# Convert data into arrays: shape (n_metrics, n_algorithms)
mean_values = np.array([[means[alg][i] for alg in algorithms] for i in range(4)])
std_values = np.array([[stds[alg][i] for alg in algorithms] for i in range(4)])

n_metrics = len(metrics)
n_algorithms = len(algorithms)
bar_width = 0.2
index = np.arange(n_metrics)

# Plot
fig, ax = plt.subplots(figsize=(10,6))
colors = ['#4995C6', '#EE9088', '#89C1B6']



for i in range(n_algorithms):
    ax.bar(index + i*bar_width, mean_values[:, i], bar_width,
           yerr=std_values[:, i], capsize=5,
           label=algorithms[i], color=colors[i])

# Labeling
ax.set_xlabel('Metrics')
ax.set_ylabel('Value')
ax.set_title('Comparison of Algorithms for Each Metric')

# Update x-axis tick labels with LaTeX-style subscripts
metric_labels = [r'$\mathrm{RRMSE}_t$', r'$\mathrm{RRMSE}_f$', 
                 r'$\mathrm{CC}_t$', r'$\mathrm{CC}_f$']
ax.set_xticks(index + bar_width * (n_algorithms - 1) / 2)
ax.set_xticklabels(metric_labels)

ax.legend(title="Algorithms")

plt.tight_layout()
plt.savefig('comparison_result_barchart.pdf')
plt.show()



#%%
import numpy as np
import matplotlib.pyplot as plt

# Algorithms and metrics
algorithms = ['tACSNet', 'SMA', 'AF']
metrics = ['RRMSE_t', 'RRMSE_f', 'CC_t', 'CC_f']

# Mean values
means = {
    'tACSNet': [0.2832, 0.6463, 0.6387, 0.8801],
    'SMA':     [0.3387, 0.7212, 0.6225, 0.8004],
    'AF':      [0.3481, 1.3038, 0.5890, 0.6695]
}

# Standard deviations
stds = {
    'tACSNet': [0.0874, 0.2359, 0.2030, 0.1147],
    'SMA':     [0.0939, 0.3652, 0.1272, 0.1727],
    'AF':      [0.0726, 0.8678, 0.1591, 0.2381]
}



# Convert data into arrays: shape (n_metrics, n_algorithms)
mean_values = np.array([[means[alg][i] for alg in algorithms] for i in range(4)])
std_values = np.array([[stds[alg][i] for alg in algorithms] for i in range(4)])

n_metrics = len(metrics)
n_algorithms = len(algorithms)
bar_width = 0.2
index = np.arange(n_metrics)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4995C6', '#EE9088', '#89C1B6']

for i in range(n_algorithms):
    bars = ax.bar(index + i * bar_width, mean_values[:, i], bar_width,
                  yerr=std_values[:, i], capsize=5,
                  label=algorithms[i], color=colors[i])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        y_pos = height * 0.05  # 5% height from bottom
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='white')


# Labeling with larger fonts
ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Value', fontsize=14)
ax.set_title('Comparison of Algorithms for Each Metric', fontsize=16)
ax.set_xticks(index + bar_width * (n_algorithms - 1) / 2)
ax.set_xticklabels(metrics, fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(title="Algorithms", fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.show()





#%% De-normalization

de_test_data = Test_1mA_10Hz_alpha[0,:]

start_idx = int(tacs_start_alpha)*sampling_rate-50
end_idx = start_idx + sampling_rate*60


x_data_prestim = de_test_data[0: start_idx]
x_data_stim = de_test_data[start_idx : end_idx]
x_data_poststim = de_test_data[end_idx : ]


plt.figure()
plt.subplot(311)
plt.plot(x_data_prestim)
plt.subplot(312)
plt.plot(x_data_stim)
plt.subplot(313)
plt.plot(x_data_poststim)


# load model
best_autoencoder = tf.keras.models.load_model("best_autoencoder")


def min_max_normalize(epoch):
    min_val = np.min(epoch)
    max_val = np.max(epoch)
    if max_val == min_val:
        return np.zeros_like(epoch)  # avoid division by zero
    return (epoch - min_val) / (max_val - min_val)


def tACS_removal(EEG_segment):
    # 1. initialize 'cleaned_eeg' array
    cleaned_eeg = np.zeros(np.shape(EEG_segment))
    # 2. normalize data
    normalized_eeg = min_max_normalize(EEG_segment)
    # 3. data dimension to fit tf model
    ready_eeg = normalized_eeg.reshape(1,1000,1)
    # 4. tf model inference
    encoded_layer = best_autoencoder.encoder(ready_eeg).numpy()
    decoded_layer = best_autoencoder.decoder(encoded_layer).numpy()
    decoded_layer = np.squeeze(decoded_layer)
    # 5. assign
    cleaned_eeg = decoded_layer
    return cleaned_eeg


def get_mean_var_prestim(prestim_data):
    mean_value = np.mean(prestim_data)
    std_value = np.std(prestim_data)
    max_value = np.max(prestim_data)
    min_value = np.min(prestim_data)
    return mean_value, std_value, max_value, min_value


def denormalization(cleaned_eeg, max_value, min_value):
    # denormalized_eeg = cleaned_eeg * std_value + mean_value
    denormalized_eeg = cleaned_eeg * (max_value - min_value) + min_value
    return denormalized_eeg
    
    
    
## 
segment_length = sampling_rate*2 
loop = int(len(x_data_stim)/segment_length)
mean_value, std_value, max_value, min_value = get_mean_var_prestim(x_data_prestim)
buffer = []
for i in range(loop):
    segment = x_data_stim[i*segment_length : (i+1)*segment_length]
    segment_cleaned = tACS_removal(segment)
    segment_denorm = denormalization(segment_cleaned, mean_value, std_value)
    
    buffer.append(segment_denorm)


processed_data_stim = np.concatenate(buffer)
    

denormalized = np.concatenate((x_data_prestim, np.squeeze(processed_data_stim), x_data_poststim))
    
    
    
    
plt.figure()
plt.plot(denormalized)
    
    
    
    
    
    


