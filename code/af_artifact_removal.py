# af_artifact_removal.py

import numpy as np

class AFMethod:
    def __init__(self, forgetting_factor=0.999, delta=0.01):
        """
        Initialize the Adaptive Filtering (AF) method using the Recursive Least Squares (RLS) algorithm.

        Parameters:
        - forgetting_factor: float, RLS forgetting factor lambda (close to 1 for stable tACS artifact)
        - delta: float, initial value for the inverse correlation matrix (small positive value)
        """
        self.lambda_ = forgetting_factor
        self.delta = delta

    def estimate_reference(self, contaminated_signal, tacs_freq, sampling_rate):
        t = np.arange(len(contaminated_signal)) / sampling_rate  # time vector
        X = np.column_stack([
            np.sin(2 * np.pi * tacs_freq * t),
            np.cos(2 * np.pi * tacs_freq * t)
        ])
        beta, _, _, _ = np.linalg.lstsq(X, contaminated_signal, rcond=None)
        fitted_waveform = X @ beta
        # amplitude = np.sqrt(beta[0]**2 + beta[1]**2)
        # phase = np.arctan2(beta[1], beta[0])
        min_val = np.min(fitted_waveform)
        max_val = np.max(fitted_waveform)
        reference_estimate = (fitted_waveform - min_val) / (max_val - min_val)
        
        return reference_estimate

    def apply(self, contaminated_signal, reference_signal=None, tacs_freq=None, sampling_rate=None):
        """
        Apply adaptive filtering to remove tACS artifacts from EEG signal.

        Parameters:
        - contaminated_signal: np.ndarray, 1D array (EEG+tACS contaminated signal)
        - reference_signal: np.ndarray or None, 1D array (estimate of tACS artifact). If None, estimate automatically.
        - tacs_freq: float or None, required if reference_signal is None.
        - sampling_rate: float or None, required if reference_signal is None.

        Returns:
        - cleaned_signal: np.ndarray, 1D array (artifact-reduced EEG signal)
        """
        if contaminated_signal.ndim != 1:
            raise ValueError("Input contaminated_signal must be a 1D array representing single-channel EEG data.")

        if reference_signal is None:
            if tacs_freq is None or sampling_rate is None:
                raise ValueError("If reference_signal is not provided, tacs_freq and sampling_rate must be specified.")
            reference_signal = self.estimate_reference(contaminated_signal, tacs_freq, sampling_rate)

        if reference_signal.ndim != 1:
            raise ValueError("reference_signal must be a 1D array.")
        if contaminated_signal.shape != reference_signal.shape:
            raise ValueError("contaminated_signal and reference_signal must have the same length.")

        n_samples = contaminated_signal.shape[0]

        # Initialize
        n_weights = 1  # Single tap adaptive filter
        w = np.zeros(n_weights)
        P = (1.0 / self.delta) * np.eye(n_weights)
        cleaned_signal = np.zeros(n_samples)

        for n in range(n_samples):
            x_n = np.array([reference_signal[n]])  # current reference input
            d_n = contaminated_signal[n]           # current contaminated input

            # Prediction
            y_n = np.dot(w, x_n)
            e_n = d_n - y_n  # error

            # RLS update
            Pi_x = np.dot(P, x_n)
            k_n = Pi_x / (self.lambda_ + np.dot(x_n.T, Pi_x))
            w = w + k_n * e_n
            P = (P - np.outer(k_n, Pi_x)) / self.lambda_

            # Save cleaned output (error signal)
            cleaned_signal[n] = e_n
            
            
        def fix_small_edge_artifacts(signal, edge_len=3):
            """
            Replace the first and last `edge_len` points of the signal
            by the nearest reliable neighbor value to eliminate small edge artifacts.
            """
            n = len(signal)
            if edge_len * 2 >= n:
                return signal  # skip if too short
        
            fixed = signal.copy()
            # Replace start edge with first reliable sample
            fixed[:edge_len] = signal[edge_len]
            # Replace end edge with last reliable sample
            fixed[-edge_len:] = signal[-edge_len-1]
        
            return fixed
        
        
        cleaned_signal = fix_small_edge_artifacts(cleaned_signal, edge_len=3)

        

        return cleaned_signal

# Example usage (remove when importing elsewhere)
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     fs = 500
#     time = np.linspace(0, 10, fs*10)
#     tacs_artifact = np.sin(2*np.pi*10*time)
#     true_eeg = 0.2*np.sin(2*np.pi*8*time) + 0.1*np.random.randn(len(time))
#     contaminated = true_eeg + tacs_artifact

#     af = AFMethod(forgetting_factor=0.999)
#     cleaned = af.apply(contaminated_signal=contaminated, tacs_freq=10, sampling_rate=fs)

#     plt.figure(figsize=(12,6))
#     plt.plot(time, contaminated, label='Contaminated EEG')
#     plt.plot(time, cleaned, label='Cleaned EEG (AF)', alpha=0.7)
#     plt.legend()
#     plt.xlabel('Time (s)')
#     plt.title('Adaptive Filtering Artifact Removal Example')
#     plt.show()



# from af_artifact_removal import AFMethod

# af = AFMethod(forgetting_factor=0.999)

# # If you DON'T have a reference signal:
# cleaned_signal = af.apply(contaminated_signal, tacs_freq=10, sampling_rate=500)

# If you DO have a reference signal:
# cleaned_signal = af.apply(contaminated_signal, reference_signal=my_reference)
