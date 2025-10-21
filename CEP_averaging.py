# -*- coding: utf-8 -*-
"""

@author: marti
"""

import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# Specify the full path to your EDF file
file_path = r"C:\Users\marti\Documents\HSJD\Cerebellum Evoked Potentials\Data\7.edf"  # Change this to your file path

# Open and read the EDF file
f = pyedflib.EdfReader(file_path)

# Get number of signals in the file
n_signals = f.signals_in_file

# Get signal labels (channel names)
signal_labels = f.getSignalLabels()

# Initialize lists BEFORE the loop
signals = []
sample_frequencies = []
units = []

# Read all signals and their metadata
for i in range(n_signals):
    signals.append(f.readSignal(i))
    sample_frequencies.append(f.getSampleFrequency(i))
    units.append(f.getPhysicalDimension(i))

# Close the file
f.close()

# Print file information
print(f"Number of signals: {n_signals}")
print(f"\nSignal Information:")
for i in range(n_signals):
    print(f"  {i+1}. {signal_labels[i]}")
    print(f"     - Sample Rate: {sample_frequencies[i]} Hz")
    print(f"     - Units: {units[i]}")
    print(f"     - Duration: {len(signals[i])/sample_frequencies[i]:.2f} seconds")

# Create plots
fig, axes = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))

# Handle single signal case
if n_signals == 1:
    axes = [axes]

# Plot each signal
for i in range(n_signals):
    # Create time array
    time = np.arange(len(signals[i])) / sample_frequencies[i]*1000
    
    # Plot the signal
    line = axes[i].plot(time, signals[i], linewidth=0.5)
    axes[i].set_ylabel(f'{signal_labels[i]} ({units[i]})')
    axes[i].set_xlabel('Time (ms)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_title(f'{signal_labels[i]} - Sample Rate: {sample_frequencies[i]} Hz - Units: {units[i]}')
    
    # Add interactive cursor
    mplcursors.cursor(line, hover=True)

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