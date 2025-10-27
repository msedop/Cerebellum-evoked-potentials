# -*- coding: utf-8 -*-
"""

@author: marti
"""

import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import mne
import math
import os

# Specify the full path to your EDF file
file_path = r"C:\Users\marti\OneDrive\Documents\HSJD\Cerebelo\Chimula Mark\13.14_stim.edf"  # Change this to your file path

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
raw.drop_channels( ['Chin+','ECG+','DI+','DD+','RONQ+','CAN+','TER+','PCO2+','EtCO+','Pos+','Tor+','Abd+','TIBI+','TIBD+','thor+','abdo+','PULS+','BEAT+','SpO2+','MKR+'])
raw.drop_channels([
    'EEG Fp1', 'EEG Fp2', 'EEG EOGI', 'EEG T3', 'EEG T4',
    'EEG O1', 'EEG EOGD', 'EEG O2', 'EEG A1', 'EEG A2', 'EEG EOGX'
])
data = raw.get_data()  # shape (n_channels, n_samples)
labels = raw.ch_names
sfreq = float(raw.info['sfreq'])
signals = [data[i].astype(np.float64) for i in range(data.shape[0])]
sample_frequencies = [sfreq for _ in labels]
units = ['' for _ in labels]


# ------------------ Filtering -------------------------------------------------
l_freq = 0.3                  # high-pass cutoff (Hz); set None to disable
h_freq = 70.0                 # low-pass cutoff (Hz); set None to disable

# Apply bandpass
raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, picks=None, fir_design='firwin', verbose=False)

# Get filtered data and optionally convert to µV (MNE typically returns Volts)
filt_data = raw_filtered.get_data()  # shape: (n_eeg, n_samples) in µV
filt_signals = [filt_data[i].astype(np.float64) for i in range(filt_data.shape[0])]

raw_data = raw.get_data()
raw_signals = [raw_data[i].astype(np.float64) for i in range(raw_data.shape[0])]

# Use the 2D array 'data' directly (n_channels, n_samples)
n_signals, n_samples = raw_data.shape
duration_mins = n_samples / (sfreq * 60)

print(f"Sampling frequency: {sfreq} Hz")
print(f"Number of EEG channels: {n_signals}")
print(f"Recording duration: {duration_mins:.2f} min\n")

# -----------------------------Re-referencing------------------------------------
'''
def reref_to_channel_numpy(data_uv, channel_labels, ref_name):
    """
    data_uv: ndarray (n_ch, n_samples) in µV
    channel_labels: list of labels length n_ch
    ref_name: channel label to use as reference (string)
    returns: data_reref (same shape) = data_uv - reference_channel
    """
    if ref_name not in channel_labels:
        raise ValueError(f"Reference channel {ref_name} not found in channel_labels")
    ref_idx = channel_labels.index(ref_name)
    ref_signal = data_uv[ref_idx:ref_idx+1, :]   # shape (1, n_samples)
    return data_uv - ref_signal                 # broadcasting subtracts ref from every channel

# usage:
filt_data = reref_to_channel_numpy(filt_data, eeg_labels, ref_name='EEG Pz')
raw_data = reref_to_channel_numpy(raw_data, eeg_labels, ref_name='EEG Pz')
'''
# ----------------------- Rewriting labels to same event ---------------------------------

# set the unified label you want for all events
new_label = 'STIM'   # choose any string you prefer (no spaces recommended)

# get existing annotations (onset, duration, description, orig_time)
anns = raw_filtered.annotations
print("Original unique annotation descriptions:", np.unique(anns.description))

# create new Annotations object with the same onsets/durations but unified descriptions
unified_descriptions = [new_label] * len(anns)
new_anns = mne.Annotations(onset=anns.onset,
                           duration=anns.duration,
                           description=unified_descriptions,
                           orig_time=anns.orig_time)

# attach to raw_filtered (overwrites previous annotations)
raw_filtered.set_annotations(new_anns)
print("Rewrote annotations. New unique descriptions:", np.unique(raw_filtered.annotations.description))

# now generate events / event_id from the new annotations
events, event_id = mne.events_from_annotations(raw_filtered)
print("event_id mapping:", event_id)
print("First 10 events:\n", events[:10])

# Launch interactive viewer with events displayed
# duration controls how many seconds are visible at once
raw_filtered.plot(n_channels=20, duration=30, scalings='auto', events=events, event_id=event_id)

#-------------------------- Rename events ------------------------------------

# --- create 0.5-second epochs starting at each event onset ---

# desired epoch length in seconds
epoch_tmin = 0.0
epoch_tmax = 1

print(f"Requested events: {len(events)}. Example events (first 10 rows):\n{events[:10]}")
print(f"Event ID mapping: {event_id}")


# Create epochs: start at event (tmin=0.0) and last 4 seconds (tmax=4.0)
epochs = mne.Epochs(raw_filtered, events, event_id=event_id,
                    tmin=epoch_tmin, tmax=epoch_tmax, baseline=None, preload=True, verbose=False)


# epoch data shape: (n_epochs, n_channels, n_times)
edata = epochs.get_data()
n_epochs, n_channels, n_times = edata.shape
epoch_duration_s = (n_times - 1) / sfreq  # approximate
print(f"Epochs shape: {edata.shape} (n_epochs, n_channels, n_times). epoch_duration ≈ {epoch_duration_s:.3f} s")



# If you want epoch data in microvolts (µV):
edata_uV = edata * 1e6

# Quick checks / visualization
#  - show average evoked across epochs for first event type (if multiple)
first_ev_key = list(event_id.keys())[0]
print(f"Plotting average for event: {first_ev_key}")
evoked = epochs[first_ev_key].average()
evoked.plot(spatial_colors=True)   # interactive MNE plot


# ----------------- Per channel average ------------------------------

# --- get per-channel mean across epochs ---
# prefer edata if available, otherwise pull from epochs
if 'edata' in globals() and edata.ndim == 3:
    # edata shape: (n_epochs, n_channels, n_times)
    mean_per_channel = np.nanmean(edata, axis=0)   # shape (n_channels, n_times)
elif 'epochs' in globals():
    mean_per_channel = np.nanmean(epochs.get_data(), axis=0)
else:
    raise RuntimeError("No epoch data found: provide edata (n_epochs,n_ch,n_times) or an epochs object.")

# --- get time vector and channel names ---
if 'epochs' in globals():
    times = epochs.times
    ch_names = list(epochs.ch_names)
else:
    # fallback: try aligned_times or kept_names
    times = aligned_times if 'aligned_times' in globals() else np.arange(mean_per_channel.shape[1]) / float(sfreq)
    ch_names = kept_names if 'kept_names' in globals() else [f"ch{i}" for i in range(mean_per_channel.shape[0])]

n_ch, n_times = mean_per_channel.shape
assert n_ch == len(ch_names), f"Channel name count ({len(ch_names)}) != data channels ({n_ch})"

# --- convert to µV if values are in Volts (heuristic) ---
# If the max absolute value is small (<1e-2) assume Volts and convert to µV
if np.nanmax(np.abs(mean_per_channel)) < 1e-2:
    mean_uV = mean_per_channel * 1e6
else:
    mean_uV = mean_per_channel.copy()
# now mean_uV is (n_channels, n_times) in µV

# --- plotting: one subplot per channel (6 x 4 grid for 24 channels) ---
n_cols = 6
n_rows = int(math.ceil(n_ch / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

for i in range(n_ch):
    ax = axes[i]
    ax.plot(times, mean_uV[i], color='C0', lw=1)
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)   # event onset / alignment
    ax.set_title(ch_names[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times[0], times[-1])

for ax in axes[n_ch:]:
    ax.axis('off')

fig.suptitle("Per-channel mean across all epochs (µV)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# =================== Detection of stimulus onset =========================

def find_stimulus_onset_per_epoch_v2(
    edata_uV,
    sfreq,
    ch_names,
    method='template_matching',
    use_best_channels=True,
    num_best_channels=5,
    template_duration_ms=1000,
    verbose=False
):
    """
    Improved stimulus onset detection using:
    1. Channel selection (use only channels with strong stimulus response)
    2. Template matching or cross-correlation based detection
    3. More robust consensus across channels
    
    Parameters
    ----------
    edata_uV : ndarray
        Epoch data in microvolts (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    ch_names : list
        Channel names
    method : str
        'template_matching' or 'amplitude_envelope'
    use_best_channels : bool
        If True, automatically select channels with strongest stimulus response
    num_best_channels : int
        Number of best channels to use for consensus
    template_duration_ms : float
        Duration of stimulus artifact to analyze (ms)
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    stimulus_onsets : ndarray
        Sample index of stimulus onset for each epoch
    reliability_scores : ndarray
        Confidence score (0-1) for each epoch's detection
    """
    
    from scipy.signal import correlate, hilbert
    from scipy.ndimage import uniform_filter1d
    
    n_epochs, n_channels, n_times = edata_uV.shape
    template_samples = int(np.ceil(template_duration_ms * sfreq / 1000))
    
    stimulus_onsets = np.zeros(n_epochs, dtype=int)
    reliability_scores = np.zeros(n_epochs)
    channel_onsets = np.zeros((n_epochs, n_channels), dtype=int)
    
    if verbose:
        print(f"Stimulus onset detection (v2):")
        print(f"  Method: {method}")
        print(f"  Using best {num_best_channels} channels: {use_best_channels}")
    
    # Step 1: Identify best channels (strongest stimulus response)
    if use_best_channels:
        if verbose:
            print(f"\n  Selecting best channels...")
        
        # Calculate signal energy/variance in first 100ms (where stimulus should be)
        early_window = int(np.ceil(0.1 * sfreq))  # first 100ms
        channel_energies = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # RMS across all epochs in early window
            channel_energies[ch] = np.sqrt(np.mean(edata_uV[:, ch, :early_window]**2))
        
        # Sort and select top channels
        best_ch_indices = np.argsort(channel_energies)[-num_best_channels:]
        best_ch_indices = np.sort(best_ch_indices)
        
        if verbose:
            for idx in best_ch_indices:
                print(f"    {ch_names[idx]}: energy={channel_energies[idx]:.1f} µV")
    else:
        best_ch_indices = np.arange(n_channels)
    
    # Step 2: Process each epoch
    for epoch_idx in range(n_epochs):
        epoch_data = edata_uV[epoch_idx, :, :]
        
        # Get detections from best channels
        onsets_this_epoch = []
        correlations_this_epoch = []
        
        for ch_idx in best_ch_indices:
            signal = epoch_data[ch_idx, :].astype(np.float64)
            
            if method == 'amplitude_envelope':
                # Use analytic signal (Hilbert transform)
                analytic_signal = np.abs(hilbert(signal))
                
                # Smooth and find onset as first significant rise
                window = int(np.ceil(2 * sfreq / 1000))  # 2ms smoothing
                if window > 1:
                    smoothed = uniform_filter1d(analytic_signal, size=window)
                else:
                    smoothed = analytic_signal
                
                # Find onset: first crossing of 40th percentile
                threshold = np.percentile(smoothed, 40)
                candidates = np.where(smoothed > threshold)[0]
                
                if len(candidates) > 0:
                    onset = candidates[0]
                else:
                    onset = np.argmax(analytic_signal)
                
                correlation = np.max(analytic_signal[:template_samples]) / (np.mean(analytic_signal) + 1e-10)
            
            elif method == 'template_matching':
                # Create a high-pass filtered version to emphasize the spike
                from scipy.signal import butter, sosfilt
                
                # Design a high-pass filter at 300 Hz (assuming stimulus is very sharp)
                sos = butter(4, 300, 'hp', fs=sfreq, output='sos')
                signal_hp = sosfilt(sos, signal)
                
                # Find the maximum of the first derivative (steepest rising edge)
                gradient = np.gradient(signal_hp)
                
                # Smooth gradient with 2ms window
                window = max(1, int(np.ceil(2 * sfreq / 1000)))
                kernel = np.ones(window) / window
                gradient_smooth = np.convolve(gradient, kernel, mode='same')
                
                # Find onset: maximum positive gradient in first 100ms
                search_window = min(int(0.1 * sfreq), n_times)
                onset = np.argmax(gradient_smooth[:search_window])
                
                # Calculate correlation as signal peak in first template window
                correlation = np.max(np.abs(signal_hp[:template_samples])) / (np.std(signal_hp) + 1e-10)
            
            onsets_this_epoch.append(onset)
            correlations_this_epoch.append(correlation)
            channel_onsets[epoch_idx, ch_idx] = onset
        
        # Consensus: use weighted median (channels with higher correlation get more weight)
        correlations_array = np.array(correlations_this_epoch)
        onsets_array = np.array(onsets_this_epoch)
        
        # Normalize correlations to 0-1 range as weights
        weights = (correlations_array - np.min(correlations_array)) / (np.max(correlations_array) - np.min(correlations_array) + 1e-10)
        
        # Weighted median
        sorted_idx = np.argsort(onsets_array)
        sorted_onsets = onsets_array[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        median_weight = cumsum_weights[-1] / 2
        median_idx = np.searchsorted(cumsum_weights, median_weight)
        median_onset = sorted_onsets[median_idx]
        
        stimulus_onsets[epoch_idx] = int(median_onset)
        
        # Reliability: inverse of normalized std dev of detections
        onset_std = np.std(onsets_array)
        reliability = np.exp(-onset_std / (np.mean(onsets_array) + 1e-10))
        reliability_scores[epoch_idx] = reliability
        
        if verbose and epoch_idx < 5:
            print(f"\n  Epoch {epoch_idx}:")
            print(f"    Channel onsets: {onsets_array} samples")
            print(f"    Consensus: {int(median_onset)} samples ({median_onset/sfreq*1000:.2f} ms)")
            print(f"    Reliability: {reliability:.2f}")
    
    if verbose:
        print(f"\n✓ Detection complete:")
        print(f"  Mean reliability: {np.mean(reliability_scores):.3f}")
        print(f"  Low reliability epochs (<0.5): {np.sum(reliability_scores < 0.5)}")
    
    return stimulus_onsets, reliability_scores, channel_onsets


# ==================== Realign Epochs to stimulus onset =======================

def realign_epochs_to_stimulus(
    edata_uV,
    stimulus_onsets,
    sfreq,
    tmin_new=-0.01,
    tmax_new=0.49,
    verbose=False
):
    """
    Realign epochs to stimulus onset and crop to desired time window.
    
    Parameters
    ----------
    edata_uV : ndarray
        Original epoch data in microvolts (n_epochs, n_channels, n_times)
    stimulus_onsets : ndarray
        Sample index of stimulus onset for each epoch (shape: n_epochs)
    sfreq : float
        Sampling frequency in Hz
    tmin_new : float
        New epoch start time relative to stimulus onset (in seconds)
    tmax_new : float
        New epoch end time relative to stimulus onset (in seconds)
    verbose : bool
        Print diagnostic information
    
    Returns
    -------
    edata_realigned : ndarray
        Realigned epoch data with potentially different time axis
    new_times : ndarray
        Time vector for realigned epochs (relative to stimulus onset)
    valid_epochs : ndarray
        Boolean array indicating which epochs could be fully realigned
        (some may be truncated at the edges)
    """
    
    n_epochs, n_channels, n_times = edata_uV.shape
    
    # Calculate new epoch length in samples
    tmin_samples = int(np.round(tmin_new * sfreq))
    tmax_samples = int(np.round(tmax_new * sfreq))
    new_n_times = tmax_samples - tmin_samples + 1
    
    edata_realigned = np.full((n_epochs, n_channels, new_n_times), np.nan, dtype=edata_uV.dtype)
    valid_epochs = np.ones(n_epochs, dtype=bool)
    
    if verbose:
        print(f"\nRealigning epochs to stimulus onset:")
        print(f"  New epoch window: [{tmin_new:.4f}, {tmax_new:.4f}] s")
        print(f"  New epoch samples: {new_n_times}")
    
    for epoch_idx in range(n_epochs):
        stimulus_sample = stimulus_onsets[epoch_idx]
        
        # Calculate sample indices in the original epoch
        start_sample = stimulus_sample + tmin_samples
        end_sample = stimulus_sample + tmax_samples
        
        # Check if the new window is within the original epoch bounds
        if start_sample < 0 or end_sample >= n_times:
            valid_epochs[epoch_idx] = False
            if verbose and epoch_idx < 3:  # Show first few problematic epochs
                print(f"  Epoch {epoch_idx}: TRUNCATED (stim @ {stimulus_sample}, range [{start_sample}, {end_sample}])")
        
        # Extract and realign (handle edge cases with NaN padding)
        src_start = max(0, start_sample)
        src_end = min(n_times, end_sample + 1)
        dst_start = max(0, -start_sample)
        dst_end = new_n_times - max(0, end_sample - n_times + 1)
        
        edata_realigned[epoch_idx, :, dst_start:dst_end] = edata_uV[epoch_idx, :, src_start:src_end]
    
    # Generate time vector relative to stimulus onset
    new_times = np.arange(new_n_times) / sfreq + tmin_new
    
    if verbose:
        n_valid = np.sum(valid_epochs)
        print(f"  Valid epochs: {n_valid}/{n_epochs}")
    
    return edata_realigned, new_times, valid_epochs


# ================== Detection of stimulus onset ====================================

print("\n" + "="*50)
print("DETECTING STIMULUS ONSET PER EPOCH (Improved)")
print("="*50)

stimulus_onsets, reliability_scores, channel_onsets = find_stimulus_onset_per_epoch_v2(
    edata_uV,
    sfreq,
    list(epochs.ch_names),
    method='amplitude_envelope',  # Try this first
    use_best_channels=True,
    num_best_channels=5,
    verbose=True
)

print(f"\nStimulus onset statistics:")
print(f"  Mean: {np.mean(stimulus_onsets/sfreq)*1000:.2f} ms")
print(f"  Std: {np.std(stimulus_onsets/sfreq)*1000:.2f} ms")
print(f"  Min: {np.min(stimulus_onsets/sfreq)*1000:.2f} ms")
print(f"  Max: {np.max(stimulus_onsets/sfreq)*1000:.2f} ms")

# Plot reliability
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(reliability_scores)), reliability_scores, color=['red' if r < 0.5 else 'green' for r in reliability_scores])
ax.axhline(0.5, color='orange', linestyle='--', label='Reliability threshold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reliability Score')
ax.set_title('Stimulus Detection Reliability per Epoch')
ax.legend()
plt.tight_layout()
plt.show()


# ================== FILTER GOOD EPOCHS ==================
print("\n" + "="*50)
print("FILTERING GOOD EPOCHS")
print("="*50)

# Create mask for good epochs (reliability >= 0.5)
good_epochs_mask = reliability_scores >= 0.5
n_good_epochs = np.sum(good_epochs_mask)

print(f"Total epochs: {len(reliability_scores)}")
print(f"Good epochs (reliability ≥ 0.5): {n_good_epochs}")
print(f"Poor epochs (reliability < 0.5): {len(reliability_scores) - n_good_epochs}")

# ================== REALIGN TO STIMULUS ONSET ==================
print("\n" + "="*50)
print("REALIGNING EPOCHS TO STIMULUS ONSET")
print("="*50)

# Realign epochs so they start at stimulus onset
tmin_realigned = 0.0      # Start AT stimulus onset
tmax_realigned = 1     # 490 ms after stimulus

edata_realigned, times_realigned, valid_epochs = realign_epochs_to_stimulus(
    edata_uV,
    stimulus_onsets,
    sfreq,
    tmin_new=tmin_realigned,
    tmax_new=tmax_realigned,
    verbose=True
)

print(f"Realigned data shape: {edata_realigned.shape}")
print(f"Time range: [{times_realigned[0]:.4f}, {times_realigned[-1]:.4f}] s")

# ================== CROP STIMULUS ARTIFACT ==================
print("\n" + "="*50)
print("CROPPING STIMULUS ARTIFACT")
print("="*50)

# Define stimulus artifact duration
crop_duration_ms = 60 # should be 10.5
crop_duration_s = crop_duration_ms / 1000.0
crop_samples = int(np.round(crop_duration_s * sfreq))

print(f"Removing stimulus artifact: {crop_duration_ms} ms ({crop_samples} samples)")

# Remove the first N samples (stimulus artifact period)
# After removal: data starts at ~10.5 ms after stimulus onset
edata_cropped_all = edata_realigned[:, :, crop_samples:]
times_cropped_all = times_realigned[crop_samples:]

print(f"After cropping:")
print(f"  Data shape: {edata_cropped_all.shape}")
print(f"  Time range: [{times_cropped_all[0]:.4f}, {times_cropped_all[-1]:.4f}] s")

# ================== SELECT ONLY GOOD EPOCHS ==================
print("\n" + "="*50)
print("SELECTING GOOD EPOCHS")
print("="*50)

# Select only good epochs
edata_good = edata_cropped_all[good_epochs_mask]
stimulus_onsets_good = stimulus_onsets[good_epochs_mask]
reliability_scores_good = reliability_scores[good_epochs_mask]

print(f"Final data shape (good epochs only): {edata_good.shape}")
print(f"  Epochs: {edata_good.shape[0]}")
print(f"  Channels: {edata_good.shape[1]}")
print(f"  Samples per epoch: {edata_good.shape[2]}")
print(f"  Epoch duration: {(edata_good.shape[2] - 1) / sfreq * 1000:.1f} ms")
print(f"\nMean reliability (good epochs): {np.mean(reliability_scores_good):.3f}")

# ================== COMPUTE EVOKED POTENTIAL ==================
print("\n" + "="*50)
print("COMPUTING EVOKED POTENTIAL")
print("="*50)

# Average across good epochs
evoked_potential = np.mean(edata_good, axis=0)  # shape: (n_channels, n_times)

print(f"Evoked potential shape: {evoked_potential.shape}")

# ================== VISUALIZE RESULTS ==================
print("\n" + "="*50)
print("VISUALIZING RESULTS")
print("="*50)

# Get channel names for good epochs
eeg_labels = list(epochs.ch_names)

# Plot evoked potential
n_ch = evoked_potential.shape[0]
n_cols = 6
n_rows = int(np.ceil(n_ch / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

for i in range(n_ch):
    ax = axes[i]
    ax.plot(times_cropped_all, evoked_potential[i], color='C0', lw=1.5)
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset')
    ax.set_title(eeg_labels[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times_cropped_all[0], times_cropped_all[-1])
    ax.grid(True, alpha=0.3)

for ax in axes[n_ch:]:
    ax.axis('off')

fig.suptitle(f"Evoked Potential After Stimulus Cropping\n({edata_good.shape[0]} good epochs, stimulus artifact removed)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ================== PLOT COMPARISON ==================
print("\nPlotting comparison: before vs after cropping...")

# Show first channel as example
ch_idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# Before cropping
mean_before = np.mean(edata_realigned[good_epochs_mask, ch_idx, :], axis=0)
ax1.plot(times_realigned, mean_before, color='C0', lw=1.5)
ax1.axvspan(0, crop_duration_s, alpha=0.3, color='red', label=f'Removed: {crop_duration_ms} ms')
ax1.axvline(0, color='k', linestyle='--', alpha=0.6)
ax1.set_ylabel('µV')
ax1.set_title(f'{eeg_labels[ch_idx]} - Before Cropping')
ax1.legend()
ax1.grid(True, alpha=0.3)

# After cropping
mean_after = np.mean(edata_good[:, ch_idx, :], axis=0)
ax2.plot(times_cropped_all, mean_after, color='C1', lw=1.5)
ax2.axvline(0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset + artifact')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('µV')
ax2.set_title(f'{eeg_labels[ch_idx]} - After Cropping')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================== SAVE RESULTS ==================
print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

# Option 1: Save as numpy arrays
output_dir = r"C:\Users\marti\OneDrive\Documents\HSJD\Cerebelo\Chimula Mark"
np.save(f"{output_dir}/evoked_potential.npy", evoked_potential)
np.save(f"{output_dir}/edata_good.npy", edata_good)
np.save(f"{output_dir}/times_cropped.npy", times_cropped_all)
np.save(f"{output_dir}/channel_names.npy", np.array(eeg_labels))

print(f"✓ Saved to {output_dir}/")
print(f"  - evoked_potential.npy: shape {evoked_potential.shape}")
print(f"  - edata_good.npy: shape {edata_good.shape}")
print(f"  - times_cropped.npy: shape {times_cropped_all.shape}")

# Option 2: Create a summary report
summary = {
    'total_epochs': len(reliability_scores),
    'good_epochs': n_good_epochs,
    'poor_epochs': len(reliability_scores) - n_good_epochs,
    'sfreq': sfreq,
    'crop_duration_ms': crop_duration_ms,
    'stimulus_onset_mean_ms': np.mean(stimulus_onsets / sfreq) * 1000,
    'stimulus_onset_std_ms': np.std(stimulus_onsets / sfreq) * 1000,
    'channels': eeg_labels,
    'evoked_potential_shape': evoked_potential.shape,
    'time_range_s': [float(times_cropped_all[0]), float(times_cropped_all[-1])],
}



# ================== GROUP AVERAGING ==================
print("\n" + "="*50)
print("AVERAGING EPOCHS IN GROUPS OF 10")
print("="*50)

group_size = 10
n_good_epochs = edata_good.shape[0]
n_groups = int(np.floor(n_good_epochs / group_size))
remainder = n_good_epochs % group_size

print(f"Total good epochs: {n_good_epochs}")
print(f"Group size: {group_size}")
print(f"Number of complete groups: {n_groups}")
print(f"Remaining epochs: {remainder}")

# Create grouped averages
grouped_averages = []
group_info = []

for group_idx in range(n_groups):
    start_idx = group_idx * group_size
    end_idx = start_idx + group_size
    
    # Average this group
    group_data = edata_good[start_idx:end_idx]
    group_mean = np.mean(group_data, axis=0)  # shape: (n_channels, n_times)
    
    grouped_averages.append(group_mean)
    group_info.append({
        'group_num': group_idx + 1,
        'start_epoch': start_idx,
        'end_epoch': end_idx,
        'n_epochs': group_size,
        'mean_reliability': np.mean(reliability_scores_good[start_idx:end_idx])
    })
    
    print(f"  Group {group_idx + 1}: epochs {start_idx}-{end_idx-1} (reliability: {group_info[-1]['mean_reliability']:.3f})")

# Handle remaining epochs if any
if remainder > 0:
    start_idx = n_groups * group_size
    end_idx = n_good_epochs
    
    group_data = edata_good[start_idx:end_idx]
    group_mean = np.mean(group_data, axis=0)
    
    grouped_averages.append(group_mean)
    group_info.append({
        'group_num': n_groups + 1,
        'start_epoch': start_idx,
        'end_epoch': end_idx,
        'n_epochs': remainder,
        'mean_reliability': np.mean(reliability_scores_good[start_idx:end_idx])
    })
    
    print(f"  Group {n_groups + 1} (incomplete): epochs {start_idx}-{end_idx-1} ({remainder} epochs, reliability: {group_info[-1]['mean_reliability']:.3f})")

grouped_averages = np.array(grouped_averages)  # shape: (n_groups, n_channels, n_times)
print(f"\nGrouped averages shape: {grouped_averages.shape}")

# ================== GENERATE PLOTS FOR EACH GROUP ==================
print("\n" + "="*50)
print("GENERATING PLOTS FOR EACH GROUP")
print("="*50)

n_channels = grouped_averages.shape[1]
n_cols = 6
n_rows = int(np.ceil(n_channels / n_cols))

for group_idx in range(len(grouped_averages)):
    group_mean = grouped_averages[group_idx]
    info = group_info[group_idx]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
    axes = axes.ravel()
    
    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        ax.plot(times_cropped_all, group_mean[ch_idx], color='C0', lw=1.5)
        ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
        ax.set_title(eeg_labels[ch_idx], fontsize=8)
        if ch_idx % n_cols == 0:
            ax.set_ylabel('µV')
        ax.set_xlim(times_cropped_all[0], times_cropped_all[-1])
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for ax in axes[n_channels:]:
        ax.axis('off')
    
    # Create informative title
    title = (f"Group {info['group_num']}: Epochs {info['start_epoch']}-{info['end_epoch']-1} "
             f"(n={info['n_epochs']}, reliability={info['mean_reliability']:.3f})")
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

print(f"\n✓ Generated {len(grouped_averages)} plots")

# ================== SUMMARY COMPARISON ==================
print("\n" + "="*50)
print("SUMMARY: ALL GROUPS COMPARISON")
print("="*50)

# Create a summary table
print("\nGroup Summary:")
print("-" * 80)
print(f"{'Group':<8} {'Epochs':<15} {'N':<5} {'Mean Reliability':<18}")
print("-" * 80)
for info in group_info:
    epoch_range = f"{info['start_epoch']}-{info['end_epoch']-1}"
    print(f"{info['group_num']:<8} {epoch_range:<15} {info['n_epochs']:<5} {info['mean_reliability']:<18.3f}")
print("-" * 80)

# ================== PLOT ALL GROUPS ON SAME FIGURE (Optional) ==================
print("\n" + "="*50)
print("CREATING COMPARISON OVERLAY (All Groups)")
print("="*50)

# Create a figure showing all groups overlaid for each channel
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

# Define colors for different groups
colors = plt.cm.tab20(np.linspace(0, 1, len(grouped_averages)))

for ch_idx in range(n_channels):
    ax = axes[ch_idx]
    
    # Plot each group
    for group_idx, group_mean in enumerate(grouped_averages):
        label = f"Group {group_idx + 1}"
        ax.plot(times_cropped_all, group_mean[ch_idx], 
                color=colors[group_idx], lw=1.5, label=label, alpha=0.7)
    
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
    ax.set_title(eeg_labels[ch_idx], fontsize=8)
    if ch_idx % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times_cropped_all[0], times_cropped_all[-1])
    ax.grid(True, alpha=0.3)

# Hide extra subplots
for ax in axes[n_channels:]:
    ax.axis('off')

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(grouped_averages))

fig.suptitle("All Groups Overlay - Per Channel Average", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# ================== STATISTICAL ANALYSIS ==================
print("\n" + "="*50)
print("STATISTICAL ANALYSIS ACROSS GROUPS")
print("="*50)

# Calculate peak amplitude for each group and channel
peak_amplitudes = np.zeros((len(grouped_averages), n_channels))

for group_idx, group_mean in enumerate(grouped_averages):
    for ch_idx in range(n_channels):
        # Find peak in a reasonable time window (e.g., 0.01-0.3s after stimulus)
        time_mask = (times_cropped_all >= 0.01) & (times_cropped_all <= 0.3)
        peak_amplitudes[group_idx, ch_idx] = np.max(np.abs(group_mean[ch_idx, time_mask]))

print("\nPeak Amplitudes (µV) per Group and Channel:")
print("-" * (80 + n_channels * 8))

# Header
header = "Group" + " " * 5
for ch_idx in range(n_channels):
    header += f"{eeg_labels[ch_idx]:<8}"
print(header)
print("-" * (80 + n_channels * 8))

# Data rows
for group_idx in range(len(grouped_averages)):
    row = f"G{group_idx+1:<5}"
    for ch_idx in range(n_channels):
        row += f"{peak_amplitudes[group_idx, ch_idx]:>8.1f}"
    print(row)

print("-" * (80 + n_channels * 8))

# ================== SAVE GROUP AVERAGES ==================
print("\n" + "="*50)
print("SAVING GROUP AVERAGES")
print("="*50)

output_dir = r"C:\Users\marti\OneDrive\Documents\HSJD\Cerebelo\Chimula Mark"

np.save(f"{output_dir}/grouped_averages.npy", grouped_averages)
np.save(f"{output_dir}/peak_amplitudes.npy", peak_amplitudes)

# Save group info as JSON
import json
group_info_json = {
    f"group_{info['group_num']}": {
        'epochs': f"{info['start_epoch']}-{info['end_epoch']-1}",
        'n_epochs': info['n_epochs'],
        'mean_reliability': float(info['mean_reliability'])
    }
    for info in group_info
}

with open(f"{output_dir}/group_info.json", 'w') as f:
    json.dump(group_info_json, f, indent=2)

print(f"✓ Saved to {output_dir}/")
print(f"  - grouped_averages.npy: shape {grouped_averages.shape}")
print(f"  - peak_amplitudes.npy: shape {peak_amplitudes.shape}")
print(f"  - group_info.json: group metadata")


# ===================== ALIGNMENT OF EPOCHS ============================

# -------- PARAMETERS / fallback ignore list (same as before) ----------
ignore_channels = [
    'EEG Fp1', 'EEG Fp2', 'EEG EOGI', 'EEG T3', 'EEG T4',
    'EEG O1', 'EEG EOGD', 'EEG O2', 'EEG A1', 'EEG A2', 'EEG EOGX'
]

out_fname = "concatenated_original_epochs_per_channel.npz"
save_to_disk = True   # set False to skip saving file

# -------- sanity checks ----------
if 'epochs' not in globals():
    raise RuntimeError("`epochs` object not found in the workspace. Create epochs first.")

sfreq = float(epochs.info['sfreq'])
n_epochs, n_ch_all, n_times = epochs.get_data().shape
all_ch_names = list(epochs.ch_names)

# decide which channels to keep:
if 'kept_names' in globals() and len(kept_names) > 0:
    channels_to_use = kept_names
else:
    channels_to_use = [ch for ch in all_ch_names if ch not in ignore_channels]

if len(channels_to_use) == 0:
    raise RuntimeError("No channels selected for concatenation (channels_to_use is empty).")

# get indices in the original epochs order
picks = [all_ch_names.index(ch) for ch in channels_to_use]

# select epoch data for the chosen channels
edata_orig = epochs.get_data()                       # shape: (n_epochs, n_ch_all, n_times) in Volts
edata_sel = edata_orig[:, picks, :]                 # (n_epochs, n_sel_channels, n_times)
n_epochs_sel, n_channels, n_samples = edata_sel.shape
print(f"Original epochs selection: epochs={n_epochs_sel}, channels={n_channels}, samples_per_epoch={n_samples}, sfreq={sfreq}")

# convert to µV for concatenation/plotting
edata_sel_uV = edata_sel * 1e6

# --------- concatenation: preserve block boundaries (no NaNs expected for original epochs) ----------
concatenated_keep = {}     # per-channel: epoch0 samples then epoch1 samples ...
concatenated_compact = {}  # identical here because original epochs are fixed-length (kept for API parity)
epoch_starts_preserve = {} # per-channel: start sample (in preserve concat) for each epoch
epoch_starts_compact = {}  # per-channel: start sample (in compact concat) for each epoch (same here)

for ch_idx, ch_name in enumerate(channels_to_use):
    segs = edata_sel_uV[:, ch_idx, :]      # shape (n_epochs, n_samples)
    # preserve: simple flatten in C-order: epoch0 samples, epoch1 samples, ...
    concat_preserve = segs.reshape(-1, order='C')   # length = n_epochs * n_samples
    concatenated_keep[ch_name] = concat_preserve

    # epoch start indices (preserve): epoch e starts at e * n_samples
    starts_pres = [e * n_samples for e in range(n_epochs_sel)]
    epoch_starts_preserve[ch_name] = starts_pres

    # compact: drop NaNs within each epoch; for original epochs this will be identical to preserve
    compact_list = []
    starts_comp = []
    cur = 0
    for e in range(n_epochs_sel):
        seg = segs[e]
        valid_mask = ~np.isnan(seg)
        valid_seg = seg[valid_mask]
        if valid_seg.size > 0:
            starts_comp.append(cur)
            compact_list.append(valid_seg)
            cur += valid_seg.size
        else:
            # no valid samples in this epoch for this channel
            starts_comp.append(-1)
    if len(compact_list) > 0:
        concat_compact = np.concatenate(compact_list)
    else:
        concat_compact = np.array([], dtype=segs.dtype)
    concatenated_compact[ch_name] = concat_compact
    epoch_starts_compact[ch_name] = starts_comp

# -------- time vectors ----------
# preserve: each epoch block has n_samples, total length = n_epochs * n_samples
t_preserve = np.arange(n_epochs_sel * n_samples) / sfreq   # seconds; starts at 0 = first epoch onset
# compact: per-channel time arrays (length depends if any NaNs were removed)
time_compact = {ch: np.arange(concatenated_compact[ch].size) / sfreq for ch in channels_to_use}

# -------- quick example plot for first 6 channels (compact concatenation) ----------
plot_chans = channels_to_use[:6]
n_plot = len(plot_chans)
fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.2*n_plot), sharex=True)
if n_plot == 1:
    axes = [axes]
for ax, ch in zip(axes, plot_chans):
    y = concatenated_compact[ch]
    t = time_compact[ch]
    ax.plot(t, y, color='C0', lw=0.6)
    # mark epoch boundaries in preserve representation (convert starts_preserve to time)
    for s in epoch_starts_preserve[ch]:
        ax.axvline(s / sfreq, color='0.7', linestyle=':', linewidth=0.6)
    ax.set_ylabel(ch)
axes[-1].set_xlabel('Time (s) from first epoch onset (original epochs concatenation)')
plt.tight_layout()
plt.show()

# -------- save results ----------
if save_to_disk:
    # Save dictionaries by converting to object arrays (np.savez can't directly store dict-of-arrays cleanly otherwise)
    np.savez(out_fname,
             concatenated_keep=concatenated_keep,
             concatenated_compact=concatenated_compact,
             epoch_starts_preserve=epoch_starts_preserve,
             epoch_starts_compact=epoch_starts_compact,
             channels=channels_to_use,
             sfreq=sfreq,
             t_preserve=t_preserve)
    print(f"Saved concatenated original-epoch arrays to: {out_fname}")

# Outputs you can use:
# - concatenated_keep: dict channel -> 1D np array (length = n_epochs * n_samples) in µV
# - concatenated_compact: dict channel -> 1D np array (compacted by dropping NaNs; same as preserve for fixed-length original epochs)
# - epoch_starts_preserve / epoch_starts_compact: dict channel -> list of start sample indices for each epoch
# - t_preserve: 1D time array seconds for preserve concatenation
# - time_compact: dict channel -> 1D time array seconds for compact concatenation

#  ============== Alignment with stimulus onset per epoch =================

# PARAMETERS - tune as needed
search_window = (0.0, 0.25)    # seconds relative to original epoch onset in which to search for the first peak
epoch_duration_s = 0.5         # desired epoch length after alignment (seconds)
detection_mode = 'gfp'         # 'gfp' (recommended) or specific channel name from epochs.ch_names
min_peak_amplitude = None      # optional absolute amplitude threshold (Volts); set to None to disable
pad_with_nan = True            # pad short epochs with NaN; if False, short epochs are dropped

# --- prepare data ---
edata = epochs.get_data()                    # (n_epochs, n_channels, n_times) in Volts
n_epochs, n_channels, n_times = edata.shape
orig_times = epochs.times                    # seconds (should start near 0.0)
sfreq = epochs.info['sfreq']

# convert search window to sample indices within epoch
s0 = np.searchsorted(orig_times, search_window[0])
s1 = np.searchsorted(orig_times, search_window[1], side='right')
s0 = max(0, s0)
s1 = min(n_times, max(s0 + 1, s1))
print(f"Searching for peak in samples {s0}:{s1-1} (times {orig_times[s0]:.3f}s..{orig_times[s1-1]:.3f}s)")

# detection channel index (if user requested a specific channel)
det_chan_idx = None
if detection_mode != 'gfp':
    if detection_mode in epochs.ch_names:
        det_chan_idx = epochs.ch_names.index(detection_mode)
        print(f"Using channel '{detection_mode}' (index {det_chan_idx}) for detection.")
    else:
        raise ValueError(f"Requested detection channel '{detection_mode}' not found in epochs.ch_names")

# --- detect first peak sample per epoch ---
peak_samples = np.full(n_epochs, -1, dtype=int)
peak_values = np.full(n_epochs, np.nan, dtype=float)

for e in range(n_epochs):
    if det_chan_idx is not None:
        sig = edata[e, det_chan_idx, :]
    else:
        # GFP (std across channels)
        sig = np.std(edata[e, :, :], axis=0)

    window = sig[s0:s1]
    if window.size == 0:
        continue

    # take the largest absolute deflection in search window as the stim peak
    local_idx = int(np.argmax(np.abs(window)))
    global_idx = s0 + local_idx
    peak_val = sig[global_idx]

    if (min_peak_amplitude is not None) and (abs(peak_val) < min_peak_amplitude):
        continue

    peak_samples[e] = global_idx
    peak_values[e] = peak_val

valid_mask = peak_samples >= 0
n_detected = valid_mask.sum()
print(f"Detected peaks in {n_detected}/{n_epochs} epochs.")

if n_detected == 0:
    raise RuntimeError("No peaks detected in any epoch — widen the search_window or check the data.")

# --- crop/align epochs so time 0 is the detected peak and keep epoch_duration_s seconds after peak ---
new_n_samples = int(round(epoch_duration_s * sfreq))
aligned = np.full((n_epochs, n_channels, new_n_samples), np.nan, dtype=float)
kept_epoch_indices = []

for e in range(n_epochs):
    ps = peak_samples[e]
    if ps < 0:
        continue
    available_after = n_times - ps
    if available_after <= 0:
        # nothing after peak
        continue
    copy_len = min(available_after, new_n_samples)
    aligned[e, :, :copy_len] = edata[e, :, ps:ps + copy_len]
    # decide whether to count this epoch as kept
    if copy_len == new_n_samples:
        kept_epoch_indices.append(e)
    else:
        if pad_with_nan:
            kept_epoch_indices.append(e)
        else:
            aligned[e, :, :] = np.nan  # will drop later

if not pad_with_nan:
    # drop epochs that are all NaN
    valid_epochs_mask = ~np.all(np.isnan(aligned).reshape(n_epochs, -1), axis=1)
    aligned = aligned[valid_epochs_mask]
    n_kept = aligned.shape[0]
    print(f"Dropped partially-short epochs. Kept {n_kept}/{n_epochs} epochs.")
else:
    n_kept = len(kept_epoch_indices)
    print(f"Kept (with padding) {n_kept}/{n_epochs} epochs. {n_epochs - n_kept} have no usable post-peak data.")

aligned_times = np.arange(0, new_n_samples) / sfreq   # time 0 at stimulus peak


# channels to ignore
ignore_channels = [
    'EEG Fp1', 'EEG Fp2', 'EEG EOGI', 'EEG T3', 'EEG T4',
    'EEG O1', 'EEG EOGD', 'EEG O2', 'EEG A1', 'EEG A2', 'EEG EOGX'
]

# Determine which channel-name list corresponds to aligned's channel axis.
# Prefer aligned_epochs.ch_names or epochs_kept.ch_names if available, otherwise use epochs.ch_names
if 'aligned_epochs' in globals() and hasattr(aligned_epochs, 'ch_names'):
    aligned_ch_names = aligned_epochs.ch_names
elif 'epochs_kept' in globals() and hasattr(epochs_kept, 'ch_names'):
    aligned_ch_names = epochs_kept.ch_names
else:
    # fallback to the original epochs names — this is only correct if aligned was built from epochs
    aligned_ch_names = epochs.ch_names

# Sanity check: if lengths still mismatch, try to trim or raise informative error
if aligned.shape[1] != len(aligned_ch_names):
    raise RuntimeError(
        f"Channel count mismatch: aligned has {aligned.shape[1]} channels but "
        f"aligned_ch_names has {len(aligned_ch_names)} entries. Make sure `aligned` "
        "was built from the same `epochs` object that provides these channel names."
    )

# Compute kept indices relative to aligned_ch_names (safe)
kept_indices = [i for i, ch in enumerate(aligned_ch_names) if ch not in ignore_channels]
kept_names = [aligned_ch_names[i] for i in kept_indices]

if len(kept_indices) == 0:
    raise RuntimeError("No channels left after excluding ignore_channels.")

# Select retained channels from aligned
aligned_kept = aligned[:, kept_indices, :]  # shape (n_epochs, n_kept_channels, new_n_samples)
n_kept_channels = aligned_kept.shape[1]

# --- compute per-channel averages (ignore NaNs) for retained channels only ---
channel_averages_kept = np.nanmean(aligned_kept, axis=0)   # shape (n_kept_channels, new_n_samples) in Volts
channel_averages_kept_uV = channel_averages_kept * 1e6     # µV

print(f"Computed channel_averages for {n_kept_channels} retained channels. shape: {channel_averages_kept.shape}")
print("Retained channels (in plotted order):", kept_names)

# plotting (same as before), using kept_names for titles
n_cols = 6
n_rows = int(np.ceil(n_kept_channels / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()
for ch in range(n_kept_channels):
    ax = axes[ch]
    ax.plot(aligned_times, channel_averages_kept_uV[ch], color='C0')
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
    ax.set_title(kept_names[ch], fontsize=8)
    if ch % n_cols == 0:
        ax.set_ylabel('µV')
for ax in axes[n_kept_channels:]:
    ax.axis('off')
axes[-1].set_xlabel('Time (s) (peak = 0)')
fig.suptitle('Channel-wise averages after peak alignment (retained channels only)')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Optional: build aligned_epochs_kept info from aligned_ch_names
aligned_info_kept = mne.pick_info(epochs.info if 'epochs' in globals() else aligned_epochs.info, kept_indices)
aligned_epochs_kept = mne.EpochsArray(aligned_kept, aligned_info_kept, tmin=0.0, verbose=False)
print("Created aligned_epochs_kept (MNE EpochsArray) with tmin=0.0 (peak aligned).")

# ====================== Concatenated epochs starting from stimulus onset ==============


# --- Requirements: aligned_kept (n_epochs, n_channels, n_samples), sfreq, kept_names (channel names) ---
if 'aligned_kept' not in globals():
    # try fallbacks
    if 'aligned' in globals() and 'kept_indices' in globals():
        aligned_kept = aligned[:, kept_indices, :]
    else:
        raise RuntimeError("aligned_kept not found. Run the alignment step first or provide aligned + kept_indices.")

sfreq = float(epochs.info['sfreq']) if 'epochs' in globals() else float(1.0 / (aligned_times[1] - aligned_times[0]))

n_epochs, n_ch, n_samp = aligned_kept.shape
print(f"aligned_kept shape: epochs={n_epochs}, channels={n_ch}, samples_per_epoch={n_samp}, sfreq={sfreq}")

# Convert to µV for concatenation/plotting if desired (aligned_kept currently in Volts)
aligned_kept_uV = aligned_kept * 1e6

# Containers for results
concatenated_keep_nans = {}   # channel_name -> 1D array with epoch blocks concatenated, NaNs preserved
concatenated_compact = {}     # channel_name -> 1D array with NaNs removed (only valid samples)
epoch_starts_compact = {}     # channel_name -> list of start sample indices (in compact concatenation) for each epoch
epoch_lengths = np.zeros(n_epochs, dtype=int)

# Precompute valid lengths per epoch (same across channels when using aligned, but some channels may be all-NaN in an epoch)
for e in range(n_epochs):
    # length of valid samples in epoch across channels (we'll compute per-channel later)
    # for diagnostics you can use available_after logic; here use per-channel when needed
    pass

# Build concatenations
for ch_idx in range(n_ch):
    ch_name = kept_names[ch_idx] if 'kept_names' in globals() else (aligned_epochs_kept.ch_names[ch_idx] if 'aligned_epochs_kept' in globals() else f'ch{ch_idx}')
    segs = aligned_kept_uV[:, ch_idx, :]   # shape (n_epochs, n_samp), µV with NaNs where padded
    # 1) simple flatten preserving NaNs (epoch blocks contiguous)
    concat_preserve = segs.reshape(-1, order='C')   # epoch0_samp0..epoch0_last, epoch1_samp0...
    concatenated_keep_nans[ch_name] = concat_preserve

    # 2) compact concatenation: drop NaNs per epoch, and record start indices
    compact_list = []
    starts = []
    cur_idx = 0
    for e in range(n_epochs):
        seg = segs[e]
        valid_mask = ~np.isnan(seg)
        valid_seg = seg[valid_mask]
        if valid_seg.size > 0:
            starts.append(cur_idx)
            compact_list.append(valid_seg)
            cur_idx += valid_seg.size
            epoch_lengths[e] = valid_seg.size
        else:
            # if an epoch has no valid samples for this channel we record a -1 to indicate missing
            starts.append(-1)
    if len(compact_list) > 0:
        concat_compact = np.concatenate(compact_list)
    else:
        concat_compact = np.array([], dtype=segs.dtype)
    concatenated_compact[ch_name] = concat_compact
    epoch_starts_compact[ch_name] = starts

print("Concatenation complete.")
print("Example channel keys:", list(concatenated_compact.keys())[:6])

# --- Build time vectors for plotting ---
# For preserve-NaNs concatenation the number of samples = n_epochs * n_samp
t_preserve = np.arange(n_epochs * n_samp) / sfreq   # seconds, starts at 0.0 = first epoch peak
# For compact concatenation we'll build per-channel time vector (same sampling interval)
time_compact = {}
for ch_name, data in concatenated_compact.items():
    time_compact[ch_name] = np.arange(data.size) / sfreq

# --- Quick plotting example: show first 6 channels compact concatenation ---
plot_chans = list(concatenated_compact.keys())[:6]
n_plot = len(plot_chans)
fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.2*n_plot), sharex=True)
if n_plot == 1:
    axes = [axes]
for ax, ch in zip(axes, plot_chans):
    y = concatenated_compact[ch]
    t = time_compact[ch]
    ax.plot(t, y, color='C0', lw=0.6)
    # mark epoch boundaries on compact concatenation (where available)
    for e, start in enumerate(epoch_starts_compact[ch]):
        if start >= 0:
            ax.axvline(start / sfreq, color='0.7', linestyle=':', linewidth=0.6)
    ax.set_ylabel(ch)
axes[-1].set_xlabel('Time (s) from first epoch peak (compact concatenation)')
plt.tight_layout()
plt.show()

# --- Optional: save concatenated arrays to disk (NumPy .npz) ---
out_fname = "concatenated_epochs_per_channel.npz"
np.savez(out_fname,
         concatenated_keep_nans=concatenated_keep_nans,
         concatenated_compact=concatenated_compact,
         epoch_starts_compact=epoch_starts_compact,
         sfreq=sfreq,
         aligned_times=aligned_times if 'aligned_times' in globals() else None)
print(f"Saved concatenated arrays to {out_fname}")


# ============= Groups of 10 averaged epochs =============================0

# PARAMETERS - tweak as desired
save_figs = False                # set True to save each figure to disk
out_dir = "group_figures"        # directory to save figs if save_figs True
n_cols = 6                       # number of columns per figure
figsize_per_subplot = (3.0, 2.2) # width,height per subplot
dpi = 150

# --- gather available data/metadata (defensive) ---
# times (aligned_times) for x-axis
if 'aligned_times' in globals():
    times = aligned_times
elif 'times' in globals():
    times = times
else:
    raise RuntimeError("No time vector found: provide `aligned_times` or `times` in the workspace.")

# channel names
if 'kept_names' in globals():
    ch_names = kept_names
elif 'aligned_epochs_kept' in globals() and hasattr(aligned_epochs_kept, 'ch_names'):
    ch_names = aligned_epochs_kept.ch_names
elif 'aligned_epochs' in globals() and hasattr(aligned_epochs, 'ch_names'):
    ch_names = aligned_epochs.ch_names
elif 'epochs' in globals() and hasattr(epochs, 'ch_names'):
    ch_names = epochs.ch_names
else:
    raise RuntimeError("No channel name list found. Provide `kept_names` or `aligned_epochs_kept.ch_names` or `epochs.ch_names`.")

# grouped averages in µV: prefer existing group_avgs_uV, else compute from aligned_kept/aligned using group_size=10
if 'group_avgs_uV' in globals():
    group_avgs = group_avgs_uV  # shape (n_groups, n_channels, n_samples)
else:
    # compute groups of size 10 from aligned_kept or aligned
    group_size = 10
    keep_partial = True
    if 'aligned_kept' in globals():
        data_epochs = aligned_kept * 1e6   # convert to µV
    elif 'aligned' in globals():
        data_epochs = aligned * 1e6
    else:
        raise RuntimeError("No aligned epoch array found (aligned_kept or aligned). Cannot compute group averages.")
    n_epochs_all = data_epochs.shape[0]
    if keep_partial:
        n_groups = math.ceil(n_epochs_all / group_size)
    else:
        n_groups = n_epochs_all // group_size
    group_indices = []
    for g in range(n_groups):
        s = g * group_size
        e = min(s + group_size, n_epochs_all)
        if (e - s) < group_size and not keep_partial:
            break
        group_indices.append((s, e))
    n_groups = len(group_indices)
    group_avgs = np.full((n_groups, data_epochs.shape[1], data_epochs.shape[2]), np.nan, dtype=float)
    for gi, (s, e) in enumerate(group_indices):
        group_avgs[gi] = np.nanmean(data_epochs[s:e], axis=0)

# sanity checks
n_groups, n_channels, n_samples = group_avgs.shape
if len(ch_names) != n_channels:
    # try to align channel name count; if mismatch, trim or raise
    if len(ch_names) > n_channels:
        ch_names = ch_names[:n_channels]
    else:
        raise RuntimeError(f"Channel name count ({len(ch_names)}) does not match grouped data channels ({n_channels}).")

# create output dir if requested
if save_figs:
    os.makedirs(out_dir, exist_ok=True)

# --- plot one figure per group ---
for gi in range(n_groups):
    grp = gi + 1
    title = f"Group {grp} average (epochs {group_indices[gi][0]}..{group_indices[gi][1]-1})" if 'group_indices' in globals() else f"Group {grp} average"
    n_rows = math.ceil(n_channels / n_cols)
    fig_w = n_cols * figsize_per_subplot[0]
    fig_h = n_rows * figsize_per_subplot[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True)
    axes = axes.ravel()

    for ch in range(n_channels):
        ax = axes[ch]
        y = group_avgs[gi, ch, :]
        ax.plot(times, y, color='C0', lw=1)
        ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
        ax.set_title(ch_names[ch], fontsize=8)
        if ch % n_cols == 0:
            ax.set_ylabel('µV')
        ax.set_xlim(times[0], times[-1])

    # turn off unused axes
    for ax in axes[n_channels:]:
        ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_figs:
        fname = os.path.join(out_dir, f"group_avg_{gi+1:02d}.png")
        fig.savefig(fname, dpi=dpi)
        print(f"Saved {fname}")

    plt.show()
    #plt.close(fig)

print(f"Plotted {n_groups} group figures (expected 12). Each figure contains {n_channels} channel subplots.")

# ========================== Stimulation spikes detection ====================

def detect_stimulation_events(data, sfreq,
                              threshold_uV=500.0,
                              window_ms=20,
                              min_channels_frac=0.5,
                              refractory_ms=100,
                              step_ms=None):
    """
    Detect stimulation events present simultaneously across many channels.

    Parameters
    ----------
    data : ndarray, shape (n_ch, n_samples) in µV
    sfreq : float Hz
    threshold_uV : float
        Peak-to-peak threshold (µV) in a short window to consider a channel contains a stim.
    window_ms : float
        Window width (ms) to compute p2p.
    min_channels_frac : float (0-1]
        Fraction of channels that must exceed threshold for a window to be considered a candidate event.
    refractory_ms : float
        Minimum time between consecutive events (ms) to merge close detections.
    step_ms : float or None
        Step between windows. If None, uses half the window.

    Returns
    -------
    event_samples : ndarray of ints (sample indices in original sampling)
    event_times_s : ndarray of floats (seconds)
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")

    n_ch, n_samples = data.shape
    win_samps = max(1, int(round(window_ms / 1000.0 * sfreq)))
    if step_ms is None:
        step_samps = max(1, win_samps // 2)
    else:
        step_samps = max(1, int(round(step_ms / 1000.0 * sfreq)))

    min_ch = max(1, int(np.ceil(min_channels_frac * n_ch)))

    starts = np.arange(0, n_samples - win_samps + 1, step_samps, dtype=int)
    if starts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    hits = np.zeros_like(starts, dtype=bool)
    for i, st in enumerate(starts):
        seg = data[:, st:st+win_samps]
        p2p = seg.max(axis=1) - seg.min(axis=1)
        hits[i] = (np.count_nonzero(p2p >= threshold_uV) >= min_ch)

    candidate_indices = np.nonzero(hits)[0]
    if candidate_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    centers = starts[candidate_indices] + win_samps // 2

    # Apply refractory: keep event centers separated by at least refractory_ms
    refractory_samps = max(1, int(round(refractory_ms / 1000.0 * sfreq)))
    event_samples = []
    last = -1e12
    for c in centers:
        if c - last >= refractory_samps:
            event_samples.append(int(c))
            last = c
    event_samples = np.array(event_samples, dtype=int)
    event_times_s = event_samples.astype(float) / float(sfreq)
    return event_samples, event_times_s


def plot_eeg_with_events(data, sfreq, channel_labels, event_samples,
                         title="EEG (µV) with events",
                         time_unit='minutes',
                         max_plot_points=30000,
                         stack_scale=1.5,
                         figsize=None,
                         show=True):
    """
    Plot stacked EEG channels with vertical event markers.
    data: ndarray (n_ch, n_samples) in µV
    event_samples: array of sample indices (in original sampling)
    """

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")
    n_ch, n_samples = data.shape

    # Determine downsample factor
    down = 1
    if (max_plot_points is not None) and (n_samples > max_plot_points):
        down = int(np.ceil(n_samples / float(max_plot_points)))
    plot_data = data[:, ::down]
    plot_n = plot_data.shape[1]

    # Build time vector consistent with plotted samples:
    # sample k in plot_data corresponds to original sample idx = k * down
    # time (s) = (k * down) / sfreq
    t_s = (np.arange(plot_n) * down) / sfreq
    if time_unit.lower().startswith('min'):
        t_plot = t_s / 60.0
        xlabel = "Time (minutes)"
        event_times_plot = (event_samples.astype(float) / sfreq) / 60.0
    else:
        t_plot = t_s
        xlabel = "Time (s)"
        event_times_plot = event_samples.astype(float) / sfreq

    # center channels
    centered = plot_data - np.mean(plot_data, axis=1, keepdims=True)

    # offsets based on median p2p
    p2p = np.ptp(centered, axis=1)
    finite = p2p[np.isfinite(p2p) & (p2p > 0)]
    if finite.size > 0:
        base = np.median(finite)
    else:
        base = np.max(p2p) if p2p.size > 0 else 1.0
    offset = base * float(stack_scale)
    if offset == 0 or not np.isfinite(offset):
        offset = 1.0

    offsets = np.arange(n_ch)[::-1] * offset
    stacked = centered + offsets[:, None]

    if figsize is None:
        height = max(6, n_ch * 0.25)
        figsize = (16, height)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    lines = []
    for i in range(n_ch):
        ln, = ax.plot(t_plot, stacked[i], color='k', linewidth=0.6)
        lines.append(ln)

    # Plot event vertical lines and markers (only those within plotted time range)
    # event_times_plot are in same units as t_plot
    in_range_mask = (event_times_plot >= t_plot[0]) & (event_times_plot <= t_plot[-1])
    event_times_to_plot = event_times_plot[in_range_mask]
    for et in event_times_to_plot:
        ax.axvline(et, color='r', linestyle='--', linewidth=1.1, alpha=0.9, zorder=5)
    # Add triangular markers at top of axis for visibility
    if event_times_to_plot.size > 0:
        # y position for markers just above top trace
        y_marker = offsets[0] + offset * 0.3
        ax.plot(event_times_to_plot, np.full_like(event_times_to_plot, y_marker),
                marker='v', color='r', linestyle='None', markersize=6, zorder=6)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_labels[::-1], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude (µV) (offset applied)")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_ylim(offsets[-1] - offset * 0.5, offsets[0] + offset * 0.6)

    # Optional interactive hover
    if mplcursors is not None:
        try:
            mplcursors.cursor(lines, hover=True)
        except Exception:
            pass

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax



# detect events
event_samples, event_times_s = detect_stimulation_events(
    filt_data,
    sfreq,
    threshold_uV=1000.0,
    window_ms=20,
    min_channels_frac=1,
    refractory_ms=100
)

print("Detected event sample indices:", event_samples)
print("Detected event times (s):", np.round(event_times_s, 3))

# plot raw and filtered with event markers (time in minutes)
plot_eeg_with_events(raw_data, sfreq, eeg_labels, event_samples,
                     title="Raw EEG (µV) with detected stimulation events",
                     time_unit='minutes', max_plot_points=30000, stack_scale=1.5)

plot_eeg_with_events(filt_data, sfreq, eeg_labels, event_samples,
                     title="Filtered EEG (µV) with detected stimulation events",
                     time_unit='minutes', max_plot_points=30000, stack_scale=1.5)



def create_epochs_from_events(data, event_samples, sfreq,
                              epoch_duration=0.5,
                              pad=False):
    """
    Create fixed-length epochs starting at each event sample.

    Parameters
    ----------
    data : ndarray (n_channels, n_samples)
        Continuous data in µV (channels x samples).
    event_samples : array-like of ints
        Event start sample indices (in original sampling).
    sfreq : float
        Sampling frequency in Hz.
    epoch_duration : float
        Epoch length in seconds (default 10.0).
    pad : bool
        If True, include events that extend past recording end and pad missing
        samples with np.nan. If False (default), drop events that don't have full epoch.

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, samples_per_epoch)
        Epochs array. dtype same as input. If pad True, padded values are NaN.
    valid_event_samples : ndarray, shape (n_epochs,)
        Event sample indices that were used (subset of input event_samples).
    epoch_times_s : ndarray, shape (samples_per_epoch,)
        Time vector for one epoch in seconds, starting at 0.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")
    n_ch, n_samples = data.shape

    samples_per_epoch = int(round(epoch_duration * float(sfreq)))
    if samples_per_epoch < 1:
        raise ValueError("epoch_duration too small for sampling frequency")

    event_samples = np.asarray(event_samples, dtype=int)
    epochs = []
    valid_events = []

    for e in event_samples:
        start = int(e)
        end = start + samples_per_epoch
        if start < 0:
            # skip negative starts
            continue
        if end <= n_samples:
            epoch = data[:, start:end]
            epochs.append(epoch)
            valid_events.append(start)
        else:
            if pad:
                # create NaN-padded epoch
                epoch = np.full((n_ch, samples_per_epoch), np.nan, dtype=data.dtype)
                if start < n_samples:
                    available = n_samples - start
                    epoch[:, :available] = data[:, start:n_samples]
                # else: event starts beyond end -> keep full-NaN epoch
                epochs.append(epoch)
                valid_events.append(start)
            else:
                # skip this event (not enough samples to build a full epoch)
                continue

    if len(epochs) == 0:
        # return empty array shaped suitably
        return (np.zeros((0, n_ch, samples_per_epoch), dtype=data.dtype),
                np.array([], dtype=int),
                np.arange(samples_per_epoch, dtype=float) / float(sfreq))

    epochs = np.stack(epochs, axis=0)  # shape (n_epochs, n_ch, samples_per_epoch)
    valid_event_samples = np.array(valid_events, dtype=int)
    epoch_times_s = np.arange(samples_per_epoch, dtype=float) / float(sfreq)
    return epochs, valid_event_samples, epoch_times_s


def epochs_events_no_stim(data, event_samples, sfreq,
                              epoch_duration=10.0,
                              pad=False,
                              exclude_ms=10.0):
    """
    Create fixed-length epochs starting after a short excluded period following each event.

    Parameters
    ----------
    data : ndarray (n_channels, n_samples)
        Continuous data in µV (channels x samples).
    event_samples : array-like of ints
        Event start sample indices (in original sampling; stimulation onset).
    sfreq : float
        Sampling frequency in Hz.
    epoch_duration : float
        Epoch length in seconds (default 10.0). Each epoch covers:
            [event + exclude_ms, event + exclude_ms + epoch_duration)
    pad : bool
        If True, include events that extend past recording end and pad missing
        samples with np.nan. If False (default), drop events that don't have full epoch.
    exclude_ms : float
        Milliseconds to exclude after stimulation onset (default 10.0 ms).

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, samples_per_epoch)
        Epochs array. dtype same as input. If pad True, padded values are NaN.
    valid_event_samples : ndarray, shape (n_epochs,)
        Event sample indices (original stimulation onset indices) corresponding to epochs.
    epoch_times_s : ndarray, shape (samples_per_epoch,)
        Time vector in seconds relative to stimulation onset (first value == exclude_ms/1000).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")
    n_ch, n_samples = data.shape

    samples_per_epoch = int(round(epoch_duration * float(sfreq)))
    if samples_per_epoch < 1:
        raise ValueError("epoch_duration too small for sampling frequency")

    # number of samples to skip from stim onset
    start_offset = int(round((exclude_ms / 1000.0) * float(sfreq)))
    if start_offset < 0:
        raise ValueError("exclude_ms must be non-negative")

    event_samples = np.asarray(event_samples, dtype=int)
    epochs = []
    valid_events = []

    for e in event_samples:
        # epoch begins after the excluded ms
        start = int(e) + start_offset
        end = start + samples_per_epoch
        if start < 0:
            # skip negative starts
            continue
        if end <= n_samples:
            epoch = data[:, start:end]
            epochs.append(epoch)
            valid_events.append(int(e))
        else:
            if pad:
                # create NaN-padded epoch
                epoch = np.full((n_ch, samples_per_epoch), np.nan, dtype=data.dtype)
                if start < n_samples:
                    available = n_samples - start
                    epoch[:, :available] = data[:, start:n_samples]
                # else: event starts beyond end -> keep full-NaN epoch
                epochs.append(epoch)
                valid_events.append(int(e))
            else:
                # skip this event (not enough samples to build a full epoch)
                continue

    if len(epochs) == 0:
        # return empty array shaped suitably
        return (np.zeros((0, n_ch, samples_per_epoch), dtype=data.dtype),
                np.array([], dtype=int),
                (np.arange(samples_per_epoch, dtype=float) / float(sfreq)) + (start_offset / float(sfreq)))

    epochs = np.stack(epochs, axis=0)  # shape (n_epochs, n_ch, samples_per_epoch)
    valid_event_samples = np.array(valid_events, dtype=int)
    # epoch times are relative to stimulation onset; first sample corresponds to exclude_ms
    epoch_times_s = (np.arange(samples_per_epoch, dtype=float) / float(sfreq)) + (start_offset / float(sfreq))
    return epochs, valid_event_samples, epoch_times_s




# Example helper to plot the average epoch stacked (µV)
def plot_average_epoch_stacked(epochs, epoch_times_s, channel_labels,
                               stack_scale=1.5, title="Average epoch (µV)", time_unit='s'):
    """
    epochs: ndarray (n_epochs, n_channels, n_samples)
    epoch_times_s: ndarray (n_samples,) in seconds
    channel_labels: list length n_channels
    """
    if epochs.size == 0:
        raise RuntimeError("No epochs to plot")

    # compute mean across epochs, ignoring NaNs
    avg = np.nanmean(epochs, axis=0)  # shape (n_channels, n_samples)
    n_ch, n_samples = avg.shape

    # center channels
    avg_centered = avg - np.nanmean(avg, axis=1, keepdims=True)

    # compute offset using median p2p
    p2p = np.nanmax(avg_centered, axis=1) - np.nanmin(avg_centered, axis=1)
    finite = p2p[np.isfinite(p2p) & (p2p > 0)]
    if finite.size > 0:
        base = np.median(finite)
    else:
        base = np.nanmax(p2p) if p2p.size > 0 else 1.0
    offset = base * float(stack_scale)
    if offset == 0 or not np.isfinite(offset):
        offset = 1.0

    offsets = np.arange(n_ch)[::-1] * offset
    stacked = avg_centered + offsets[:, None]

    # convert time axis if needed
    if time_unit.startswith('min'):
        t_plot = epoch_times_s / 60.0
        xlabel = "Time (minutes)"
    elif time_unit.startswith('ms'):
        t_plot = epoch_times_s * 1000.0
        xlabel = "Time (ms)"
    else:
        t_plot = epoch_times_s
        xlabel = "Time (s)"

    fig, ax = plt.subplots(figsize=(12, max(4, n_ch*0.25)))
    for i in range(n_ch):
        ax.plot(t_plot, stacked[i], color='k', linewidth=0.8)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_labels[::-1], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude (µV) (offset applied)")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_ylim(offsets[-1] - offset*0.5, offsets[0] + offset*0.5)
    plt.tight_layout()
    plt.show()
    return fig, ax


# Epochs, drop events that go beyond recording end:
epochs, valid_events, epoch_times_s = create_epochs_from_events(filt_data, event_samples, sfreq, epoch_duration=0.8, pad=False)
print(f"Built {epochs.shape[0]} epochs (shape: {epochs.shape}). Dropped {len(event_samples)-len(valid_events)} events near end.")

# If you prefer padding (keep all events but pad tails with NaN):
# epochs_padded, valid_events_padded, epoch_times_s = create_epochs_from_events(filt_data, event_samples, sfreq, epoch_duration=10.0, pad=True)

# Quick sanity checks:
n_epochs = epochs.shape[0]
samples_per_epoch = epochs.shape[2]
print("samples per epoch:", samples_per_epoch, "which equals", epoch_times_s[-1], "seconds approx")
print("Event start times (s):", np.round(valid_events.astype(float)/sfreq, 3))

# Plot average epoch across events (stacked channels) in seconds or minutes:
plot_average_epoch_stacked(epochs, epoch_times_s, eeg_labels, stack_scale=1.5, title="Average 800ms epoch (filtered)", time_unit='s')


# NO STIM EPOCHS
# Build 10 s epochs, drop events that go beyond recording end:
epochs, valid_events, epoch_times_s = epochs_events_no_stim(filt_data, event_samples, sfreq, epoch_duration=0.8, pad=False)
print(f"Built {epochs.shape[0]} epochs (shape: {epochs.shape}). Dropped {len(event_samples)-len(valid_events)} events near end.")


n_epochs = epochs.shape[0]
samples_per_epoch = epochs.shape[2]
print("samples per epoch:", samples_per_epoch, "which equals", epoch_times_s[-1], "seconds approx")
print("Event start times (s):", np.round(valid_events.astype(float)/sfreq, 3))

plot_average_epoch_stacked(epochs, epoch_times_s, eeg_labels, stack_scale=1.5, title="Average 800ms epoch epoch (filtered)", time_unit='s')
# Or in minutes:
# plot_average_epoch_stacked(epochs, epoch_times_s, eeg_labels, time_unit='min')


# ===== CREATE 10-SECOND EPOCHS =====
epoch_duration = 1  # seconds

# Create epochs for each signal
all_epochs = []
for i in range(n_signals):
    signal = filt_signals[i]
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
    
    print(f"\nSignal {i+1} ({labels[i]}):")
    print(f"  - Total epochs created: {n_epochs}")
    print(f"  - Samples per epoch: {samples_per_epoch}")
    print(f"  - Epoch duration: {epoch_duration} seconds")

# ===== PLOT EPOCHS =====
# Choose which signal to plot (0 for first signal)
signal_to_plot = 14

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
    axes[epoch_idx].set_ylabel(f'{labels[signal_to_plot]} ({units[signal_to_plot]})')
    axes[epoch_idx].set_xlabel('Time (seconds)')
    axes[epoch_idx].grid(True, alpha=0.3)
    axes[epoch_idx].set_title(f'Epoch {epoch_idx + 1} - {labels[signal_to_plot]}')
    axes[epoch_idx].set_xlim(0, epoch_duration)
    
    # Add interactive cursor
    mplcursors.cursor(line, hover=True)

plt.tight_layout()
plt.show()

# ===== SAVE EPOCHS (OPTIONAL) =====
# Uncomment to save epochs as numpy array
# np.save('epochs.npy', all_epochs)
# print("\nEpochs saved to 'epochs.npy'")
