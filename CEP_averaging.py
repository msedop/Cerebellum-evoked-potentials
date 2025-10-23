# -*- coding: utf-8 -*-
"""

@author: marti
"""

import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import mne

# Specify the full path to your EDF file
file_path = r"C:\Users\marti\OneDrive\Documents\HSJD\Cerebelo\Chimula Mark\12.33.edf"  # Change this to your file path

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
data = raw.get_data()  # shape (n_channels, n_samples)
labels = raw.ch_names
sfreq = float(raw.info['sfreq'])
signals = [data[i].astype(np.float64) for i in range(data.shape[0])]
sample_frequencies = [sfreq for _ in labels]
units = ['' for _ in labels]

n_signals = len(signals)
print(f"Number of signals: {n_signals}")
for i in range(n_signals):
    print(f"  {i+1}. {labels[i]} - Sample Rate: {sample_frequencies[i]} Hz - Duration: {len(signals[i])/sample_frequencies[i]:.2f} s")


# Select only EEG channels (case-insensitive match for "EEG")
eeg_idx = [i for i, ch in enumerate(labels) if 'EEG' in ch.upper()]
if len(eeg_idx) == 0:
    raise RuntimeError("No channels with 'EEG' in the name were found.")

# Set EEG data labels 
eeg_labels = [labels[i] for i in eeg_idx]


l_freq = 0.3                  # high-pass cutoff (Hz); set None to disable
h_freq = 70.0                 # low-pass cutoff (Hz); set None to disable

# Apply bandpass
raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, picks=None, fir_design='firwin', verbose=False)

# Get filtered data and optionally convert to µV (MNE typically returns Volts)
filt_data = raw_filtered.get_data()[eeg_idx,:] #* 1e6  # shape: (n_eeg, n_samples) in µV
filt_signals = [filt_data[i].astype(np.float64) for i in range(filt_data.shape[0])]

raw_data = raw.get_data()[eeg_idx,:]
raw_signals = [raw_data[i].astype(np.float64) for i in range(raw_data.shape[0])]

# Use the 2D array 'data' directly (n_channels, n_samples)
n_signals, n_samples = raw_data.shape
duration_mins = n_samples / (sfreq * 60)

print(f"Sampling frequency: {sfreq} Hz")
print(f"Number of channels: {n_signals}")
print(f"Recording duration: {duration_mins:.2f} min\n")

# ----------------------- Signal Visualisation ---------------------------------

# Create plots
fig, axes = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))

# Handle single signal case
if n_signals == 1:
    axes = [axes]

# Plot each signal
for i in range(n_signals):
    # Create time array
    time = np.arange(len(raw_signals[i])) / sample_frequencies[i]*1000
    
    # Plot the signal
    line = axes[i].plot(time, raw_signals[i], linewidth=0.5)
    axes[i].set_ylabel(f'{labels[i]} ({units[i]})')
    axes[i].set_xlabel('Time (ms)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_title(f'{labels[i]} - Sample Rate: {sample_frequencies[i]} Hz - Units: {units[i]}')
    
    # Add interactive cursor
    mplcursors.cursor(line, hover=True)

plt.tight_layout()
plt.show()



n_eeg, n_samples = eeg_uv.shape
t = np.arange(n_samples) / sfreq  # time in seconds

# Compute sensible offset: median peak-to-peak across EEG channels
p2p = np.ptp(eeg_uv, axis=1)
finite = p2p[np.isfinite(p2p) & (p2p > 0)]
if finite.size > 0:
    base = np.median(finite)
else:
    base = np.max(p2p) if p2p.size > 0 else 1.0
offset = base * 1.5  # multiplier to separate traces

# Build offsets so first EEG channel is on top
offsets = np.arange(n_eeg)[::-1] * offset
stacked = (eeg_uv - np.mean(eeg_uv, axis=1, keepdims=True)) + offsets[:, None]

# Plot
fig, ax = plt.subplots(figsize=(16, max(6, n_eeg * 0.25)))
lines = []
for i in range(n_eeg):
    ln, = ax.plot(t, stacked[i], color='k', linewidth=0.6)
    lines.append(ln)

# Put channel labels on the y-axis at the offset positions (reverse labels to match stack order)
ax.set_yticks(offsets)
ax.set_yticklabels(eeg_labels[::-1], fontsize=9)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV)  (offset applied)")
ax.set_title("Raw EEG channels (stacked, µV)")
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_xlim(t[0], t[-1])
ax.set_ylim(offsets[-1] - offset * 0.5, offsets[0] + offset * 0.5)

# Optional interactive hover (shows x,y)
if mplcursors is not None:
    try:
        mplcursors.cursor(lines, hover=True)
    except Exception:
        pass

plt.tight_layout()
plt.show()


# ===== CREATE 10-SECOND EPOCHS =====
epoch_duration = 0.1  # seconds

# Create epochs for each signal
all_epochs = []
for i in range(n_signals):
    signal = signals[i]
    fs = sample_frequencies[i]
    
    # Calculate samples per epoch
    samples_per_epoch = int(epoch_duration * fs)
    
    # Calculate total number of epochs
    total_samples = len(signal)
    n_epochs = int(total_samples / samples_per_epoch)
    
    # Split signal into epochs
    epochs = []
    for epoch_idx in range(n_epochs):
        start_idx = epoch_idx * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        epoch = signal[start_idx:end_idx]
        epochs.append(epoch)
    
    all_epochs.append(epochs)
    
    print(f"\nSignal {i+1} ({signal_labels[i]}):")
    print(f"  - Total epochs created: {n_epochs}")
    print(f"  - Samples per epoch: {samples_per_epoch}")
    print(f"  - Epoch duration: {epoch_duration} seconds")

# ===== PLOT EPOCHS =====
# Choose which signal to plot (0 for first signal)
signal_to_plot = 0

# Choose which epochs to plot (e.g., first 5 epochs)
epochs_to_display = min(5, len(all_epochs[signal_to_plot]))  # Display up to 5 epochs

fig, axes = plt.subplots(epochs_to_display, 1, figsize=(14, 3 * epochs_to_display))

# Handle single epoch case
if epochs_to_display == 1:
    axes = [axes]

for epoch_idx in range(epochs_to_display):
    epoch = all_epochs[signal_to_plot][epoch_idx]
    fs = sample_frequencies[signal_to_plot]
    
    # Create time array for this epoch
    time = np.arange(len(epoch)) / fs
    
    # Plot the epoch
    line = axes[epoch_idx].plot(time, epoch, linewidth=0.5)
    axes[epoch_idx].set_ylabel(f'{signal_labels[signal_to_plot]} ({units[signal_to_plot]})')
    axes[epoch_idx].set_xlabel('Time (seconds)')
    axes[epoch_idx].grid(True, alpha=0.3)
    axes[epoch_idx].set_title(f'Epoch {epoch_idx + 1} - {signal_labels[signal_to_plot]}')
    axes[epoch_idx].set_xlim(0, epoch_duration)
    
    # Add interactive cursor
    mplcursors.cursor(line, hover=True)

plt.tight_layout()
plt.show()

# ===== SAVE EPOCHS (OPTIONAL) =====
# Uncomment to save epochs as numpy array
# np.save('epochs.npy', all_epochs)
# print("\nEpochs saved to 'epochs.npy'")
