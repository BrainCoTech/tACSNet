# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:02:51 2025

@author: xingl
"""

# sma_artifact_removal.py

import numpy as np

class SMAMethod:
    def __init__(self, tacs_freq, sampling_rate, neighbor_percent=0.05):
        """
        Initialize the SMA method.

        Parameters:
        - tacs_freq: float, the frequency of the applied tACS (in Hz)
        - sampling_rate: float, the sampling rate of EEG signal (in Hz)
        - neighbor_percent: float, percentage of total segments to use as neighbors (default 5%)
        """
        self.tacs_freq = tacs_freq
        self.sampling_rate = sampling_rate
        self.neighbor_percent = neighbor_percent

        # Calculate segment length (in samples)
        samples_per_period = sampling_rate / tacs_freq
        if not samples_per_period.is_integer():
            # If not integer, use 2 periods
            self.segment_length = int(round(2 * samples_per_period))
        else:
            self.segment_length = int(samples_per_period)

    def apply(self, contaminated_signal):
        """
        Apply SMA artifact removal to the contaminated EEG signal.

        Parameters:
        - contaminated_signal: np.ndarray, shape (n_samples,)

        Returns:
        - cleaned_signal: np.ndarray, shape (n_samples,)
        """
        if contaminated_signal.ndim != 1:
            raise ValueError("Input contaminated_signal must be a 1D array representing single-channel EEG data.")

        n_samples = contaminated_signal.shape[0]
        n_segments = n_samples // self.segment_length

        if n_segments < 1:
            raise ValueError("Input data is too short for SMA processing. Please provide longer recordings.")

        M = max(1, int(np.round(self.neighbor_percent * n_segments)))

        # Split into non-overlapping segments
        segments = np.array(np.split(contaminated_signal[:n_segments*self.segment_length], n_segments))

        # Build artifact template using moving average over neighboring segments
        artifact_segments = np.zeros_like(segments)
        for n in range(n_segments):
            start_idx = max(0, n - M)
            end_idx = min(n_segments, n + M + 1)

            neighbor_segments = segments[start_idx:end_idx]
            artifact_segments[n] = np.mean(neighbor_segments, axis=0)

        # Subtract artifact template from original segments
        cleaned_segments = segments - artifact_segments

        # Reconstruct the cleaned signal
        cleaned_signal = cleaned_segments.reshape(-1)

        # Pad if necessary to match original length
        if len(cleaned_signal) < n_samples:
            cleaned_signal = np.concatenate((cleaned_signal, contaminated_signal[n_segments*self.segment_length:]))

        return cleaned_signal

# Example usage (remove when importing elsewhere)
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     fs = 500  # sampling rate
#     tacs_freq = 10  # Hz
#     time = np.linspace(0, 10, fs*10)
#     contaminated = np.sin(2*np.pi*10*time) + 0.5*np.random.randn(len(time))

#     sma = SMAMethod(tacs_freq=tacs_freq, sampling_rate=fs)
#     cleaned = sma.apply(contaminated)

#     plt.figure(figsize=(12,6))
#     plt.plot(time, contaminated, label='Contaminated EEG')
#     plt.plot(time, cleaned, label='Cleaned EEG (SMA)', alpha=0.7)
#     plt.legend()
#     plt.xlabel('Time (s)')
#     plt.title('SMA Artifact Removal Example')
#     plt.show()
