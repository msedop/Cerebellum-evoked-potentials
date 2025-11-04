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
from typing import Optional, Union, Tuple, List, Dict, Any
from matplotlib.colors import to_rgb
import pandas as pd
from tensorpac.utils import ITC, PSD

# ============================================================================
# CONFIGURATION
# ============================================================================

base_path = r"C:\Users\marti\Documents\HSJD\CCEPs\Martin Garcia"

# List of EDF files to process (in temporal order)
edf_files = [
    'CCEPs_13.10.edf',
    'CCEPs_13.14.edf',
    'CCEPs_13.20.edf'
]

# Channels to drop (non-EEG)
channels_to_drop_physio = ['Chin+','ECG+','DI+','DD+','RONQ+','CAN+','TER+','PCO2+',
                            'EtCO+','Pos+','Tor+','Abd+','TIBI+','TIBD+','thor+',
                            'abdo+','PULS+','BEAT+','SpO2+','MKR+']

# Channels to drop (unwanted EEG)
channels_to_drop_eeg = ['EEG EOGI', 'EEG O1', 'EEG EOGD', 'EEG O2', 'EEG A1', 'EEG A2']

# Channels to drop (specific)
channels_to_drop_specific = ['EEG P4', 'EEG T6']

# All channels to drop
channels_to_drop_all = channels_to_drop_physio + channels_to_drop_eeg + channels_to_drop_specific

# Filtering parameters
l_freq = 0.3   # high-pass cutoff (Hz)
h_freq = 70.0  # low-pass cutoff (Hz)

# Epoching parameters
epoch_tmin = 0.0
epoch_tmax = 0.22

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_time_from_filename(filename: str) -> Tuple[int, int]:
    """
    Extract hour and minute from filename like 'CCEPs_13.10.edf'
    Returns: (hour, minute)
    """
    base = filename.replace('.edf', '').replace('CCEPs_', '')
    parts = base.split('.')
    return int(parts[0]), int(parts[1])

def preprocess_raw(raw: mne.io.Raw, channels_to_drop: List[str]) -> mne.io.Raw:
    """
    Drop specified channels and return the preprocessed raw object.
    """
    # Drop non-EEG channels
    channels_available = raw.ch_names
    channels_to_drop_valid = [ch for ch in channels_to_drop if ch in channels_available]
    
    if channels_to_drop_valid:
        raw.drop_channels(channels_to_drop_valid)
    
    return raw

def set_standard_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Attach standard 10-20 montage to raw object.
    """
    std = mne.channels.make_standard_montage('standard_1020')
    std_pos = std.get_positions()['ch_pos']
    
    ch_pos = {}
    missing = []
    for ch in raw.ch_names:
        std_name = ch.replace('EEG ', '').strip()
        if std_name in std_pos:
            ch_pos[ch] = std_pos[std_name]
        else:
            missing.append(ch)
    
    if len(ch_pos) == 0:
        raise RuntimeError("No channels matched the standard 10-20 names. Check channel labels.")
    
    montage_subset = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage_subset)
    
    return raw, missing

def create_epochs_from_file(
    file_path: str,
    l_freq: float,
    h_freq: float,
    epoch_tmin: float,
    epoch_tmax: float,
    verbose: bool = False
) -> Tuple[mne.Epochs, np.ndarray, mne.io.Raw]:
    """
    Load an EDF file, preprocess, filter, and create epochs.
    
    Returns:
        epochs: mne.Epochs object
        edata: raw epoch data (n_epochs, n_channels, n_times)
        raw_filtered: preprocessed raw object
    """
    # Load raw data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Preprocess: drop channels
    raw = preprocess_raw(raw, channels_to_drop_all)
    
    # Set montage BEFORE filtering
    raw, missing = set_standard_montage(raw)
    
    # Filter
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, 
                                      picks=None, fir_design='firwin', verbose=False)
    
    # Get annotations and create unified events
    anns = raw_filtered.annotations
    new_label = 'STIM'
    unified_descriptions = [new_label] * len(anns)
    new_anns = mne.Annotations(onset=anns.onset,
                               duration=anns.duration,
                               description=unified_descriptions,
                               orig_time=anns.orig_time)
    raw_filtered.set_annotations(new_anns)
    
    # Get events
    events, event_id = mne.events_from_annotations(raw_filtered)
    
    if verbose:
        print(f"File: {file_path}")
        print(f"  Channels: {raw_filtered.ch_names}")
        print(f"  Events found: {len(events)}")
        if missing:
            print(f"  Missing from montage: {missing}")
        print(f"  Event ID: {event_id}\n")
    
    # Create epochs
    epochs = mne.Epochs(raw_filtered, events, event_id=event_id,
                        tmin=epoch_tmin, tmax=epoch_tmax, baseline=None, 
                        preload=True, verbose=False)
    
    edata = epochs.get_data()
    
    return epochs, edata, raw_filtered

# ============================================================================
# MAIN PROCESSING
# ============================================================================

# Sort files by time
file_info = [(f, extract_time_from_filename(f)) for f in edf_files]
file_info_sorted = sorted(file_info, key=lambda x: (x[1][0], x[1][1]))
edf_files_sorted = [f[0] for f in file_info_sorted]

print("=" * 80)
print("LOADING AND PREPROCESSING EDF FILES")
print("=" * 80)
print(f"Files to process (in temporal order):")
for idx, (fname, (hour, minute)) in enumerate(file_info_sorted):
    print(f"  {idx+1}. {fname} ({hour:02d}:{minute:02d})")
print()

# Load and process each file
all_epochs_list = []
all_raw_filtered_list = []
sfreq = None
ch_names = None
times = None

for file_idx, fname in enumerate(edf_files_sorted):
    file_path = os.path.join(base_path, fname)
    
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        continue
    
    print(f"Processing file {file_idx + 1}/{len(edf_files_sorted)}: {fname}")
    
    try:
        epochs, edata, raw_filtered = create_epochs_from_file(
            file_path, l_freq, h_freq, epoch_tmin, epoch_tmax, verbose=True
        )
        
        all_epochs_list.append(epochs)
        all_raw_filtered_list.append(raw_filtered)
        
        # Store sampling frequency and channel names from first file
        if sfreq is None:
            sfreq = float(raw_filtered.info['sfreq'])
            ch_names = list(raw_filtered.ch_names)
            times = epochs.times
        
        print(f"  ✓ Successfully loaded: {edata.shape[0]} epochs, {edata.shape[1]} channels, {edata.shape[2]} timepoints")
        print(f"  ✓ Sampling frequency: {sfreq} Hz")
        print(f"  ✓ Channels: {ch_names}\n")
        
    except Exception as e:
        print(f"  ✗ ERROR processing {fname}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        continue

# Check that all files were successfully loaded
if not all_epochs_list:
    raise RuntimeError("No files were successfully loaded!")

print(f"\nSuccessfully loaded {len(all_epochs_list)} files")

# ============================================================================
# MERGE EPOCHS ACROSS FILES
# ============================================================================

print("\n" + "=" * 80)
print("MERGING EPOCHS ACROSS FILES")
print("=" * 80)

# Concatenate all epochs objects
merged_epochs = mne.concatenate_epochs(all_epochs_list, verbose=False)

# Re-apply montage after concatenation (since it may be lost)
if all_raw_filtered_list:
    first_raw = all_raw_filtered_list[0]
    merged_epochs.set_montage(first_raw.get_montage())

# Get merged epoch data
merged_edata = merged_epochs.get_data()
n_epochs_merged, n_channels, n_times = merged_edata.shape

print(f"\nMerged epochs shape: {merged_edata.shape}")
print(f"  Total epochs: {n_epochs_merged}")
print(f"  Channels: {n_channels}")
print(f"  Timepoints per epoch: {n_times}")
print(f"  Channel names: {list(merged_epochs.ch_names)}")

# Convert to µV if needed
if np.nanmax(np.abs(merged_edata)) < 1e-2:
    merged_edata_uV = merged_edata * 1e6
else:
    merged_edata_uV = merged_edata.copy()


# ============================================================================
# ADVANCED ARTIFACT REJECTION: DETECT OSCILLATORY ARTIFACTS
# ============================================================================

print("\n" + "=" * 80)
print("ADVANCED ARTIFACT REJECTION: OSCILLATORY ARTIFACTS")
print("=" * 80)

from scipy import signal
from scipy.fft import fft, fftfreq

def detect_oscillatory_artifacts(edata, sfreq, verbose=True):
    """
    Detect epochs with strong oscillatory activity (sine-like waves).
    Uses multiple criteria:
    1. Spectral power in high-frequency bands
    2. Kurtosis (peakedness) - artifacts are more peaked/sinusoidal
    3. Zero-crossing rate - artifacts have regular zero crossings
    4. Variance - compare variance across time windows
    
    Parameters:
    -----------
    edata : ndarray (n_epochs, n_channels, n_times)
        Epoch data in µV
    sfreq : float
        Sampling frequency in Hz
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    artifact_scores : ndarray (n_epochs,)
        Artifact score for each epoch (0-1, higher = more artifact-like)
    good_epoch_indices : ndarray
        Indices of epochs with low artifact scores
    artifact_epoch_indices : ndarray
        Indices of epochs with high artifact scores
    artifact_details : dict
        Detailed breakdown of artifact metrics
    """
    from scipy.stats import kurtosis
    
    n_epochs, n_channels, n_times = edata.shape
    
    # Initialize score arrays
    spectral_scores = np.zeros(n_epochs)
    kurtosis_scores = np.zeros(n_epochs)
    zero_cross_scores = np.zeros(n_epochs)
    variance_scores = np.zeros(n_epochs)
    
    # Compute metrics for each epoch
    for ep in range(n_epochs):
        epoch_data = edata[ep]  # shape: (n_channels, n_times)
        
        # ===== METRIC 1: Spectral Power in High Frequencies =====
        # Artifacts often have concentrated power in specific narrow bands
        freqs = np.fft.fftfreq(n_times, 1/sfreq)
        pos_freqs_idx = freqs > 0
        
        spectral_powers = np.zeros((n_channels, n_times // 2))
        for ch in range(n_channels):
            fft_data = np.fft.fft(epoch_data[ch])
            power = np.abs(fft_data[pos_freqs_idx]) ** 2
            spectral_powers[ch, :len(power)] = power
        
        # Look for concentration of power (high peakedness in spectrum)
        mean_power = np.mean(spectral_powers)
        max_power = np.max(spectral_powers)
        spectral_scores[ep] = max_power / (mean_power + 1e-10)  # Peak-to-mean ratio
        
        # ===== METRIC 2: Kurtosis (Peak-like quality) =====
        # Sinusoids have lower kurtosis than normal EEG; artifacts may have distinct patterns
        kurts = np.array([kurtosis(epoch_data[ch]) for ch in range(n_channels)])
        kurtosis_scores[ep] = np.mean(kurts)
        
        # ===== METRIC 3: Zero-Crossing Rate =====
        # Regular sinusoids have very consistent zero-crossing rates
        zcr_vals = []
        for ch in range(n_channels):
            zero_crossings = np.sum(np.abs(np.diff(np.sign(epoch_data[ch])))) / 2
            zcr_vals.append(zero_crossings / n_times)
        zero_cross_scores[ep] = np.std(zcr_vals)  # Low std = regular pattern
        
        # ===== METRIC 4: Variance consistency across windows =====
        # Artifacts have consistent amplitude across time; neural response varies
        window_size = n_times // 4  # Divide epoch into 4 windows
        window_vars = []
        for w in range(4):
            start = w * window_size
            end = (w + 1) * window_size if w < 3 else n_times
            window_var = np.var(epoch_data[:, start:end])
            window_vars.append(window_var)
        
        variance_scores[ep] = 1.0 - (np.std(window_vars) / (np.mean(window_vars) + 1e-10))
    
    # Normalize scores to 0-1 range
    spectral_scores = (spectral_scores - np.min(spectral_scores)) / (np.max(spectral_scores) - np.min(spectral_scores) + 1e-10)
    kurtosis_scores = (kurtosis_scores - np.min(kurtosis_scores)) / (np.max(kurtosis_scores) - np.min(kurtosis_scores) + 1e-10)
    zero_cross_scores = (zero_cross_scores - np.min(zero_cross_scores)) / (np.max(zero_cross_scores) - np.min(zero_cross_scores) + 1e-10)
    variance_scores = (variance_scores - np.min(variance_scores)) / (np.max(variance_scores) - np.min(variance_scores) + 1e-10)
    
    # Combine scores with weights
    weights = {'spectral': 0.35, 'kurtosis': 0.25, 'zero_cross': 0.25, 'variance': 0.15}
    artifact_scores = (weights['spectral'] * spectral_scores + 
                      weights['kurtosis'] * kurtosis_scores +
                      weights['zero_cross'] * zero_cross_scores +
                      weights['variance'] * variance_scores)
    
    # Adaptive threshold using percentile (top 20% are considered artifacts)
    threshold = np.percentile(artifact_scores, 80)
    
    good_epoch_indices = np.where(artifact_scores <= threshold)[0]
    artifact_epoch_indices = np.where(artifact_scores > threshold)[0]
    
    artifact_details = {
        'spectral_scores': spectral_scores,
        'kurtosis_scores': kurtosis_scores,
        'zero_cross_scores': zero_cross_scores,
        'variance_scores': variance_scores,
        'combined_scores': artifact_scores,
        'threshold': threshold,
        'n_good': len(good_epoch_indices),
        'n_artifact': len(artifact_epoch_indices),
        'percent_good': 100 * len(good_epoch_indices) / n_epochs,
        'percent_artifact': 100 * len(artifact_epoch_indices) / n_epochs,
    }
    
    if verbose:
        print(f"\nOscillatory Artifact Detection Results:")
        print(f"  Total epochs: {n_epochs}")
        print(f"  Good epochs: {artifact_details['n_good']} ({artifact_details['percent_good']:.1f}%)")
        print(f"  Artifact epochs: {artifact_details['n_artifact']} ({artifact_details['percent_artifact']:.1f}%)")
        print(f"  Artifact score threshold: {threshold:.3f}")
        print(f"\nArtifact Score Ranges (normalized 0-1):")
        print(f"  Spectral: mean={np.mean(spectral_scores):.3f}, std={np.std(spectral_scores):.3f}")
        print(f"  Kurtosis: mean={np.mean(kurtosis_scores):.3f}, std={np.std(kurtosis_scores):.3f}")
        print(f"  Zero-crossing: mean={np.mean(zero_cross_scores):.3f}, std={np.std(zero_cross_scores):.3f}")
        print(f"  Variance consistency: mean={np.mean(variance_scores):.3f}, std={np.std(variance_scores):.3f}")
    
    return artifact_scores, good_epoch_indices, artifact_epoch_indices, artifact_details

# ============================================================================
# RUN OSCILLATORY ARTIFACT DETECTION
# ============================================================================

artifact_scores, good_indices, artifact_indices, details = detect_oscillatory_artifacts(
    merged_edata_uV,
    sfreq=sfreq,
    verbose=True
)

# ============================================================================
# VISUALIZATION 1: ARTIFACT SCORE DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Combined artifact score histogram
ax = axes[0, 0]
ax.hist(artifact_scores, bins=50, color='C0', alpha=0.7, edgecolor='black')
ax.axvline(details['threshold'], color='r', linestyle='--', linewidth=2.5, 
           label=f'Threshold ({details["threshold"]:.3f})')
ax.set_xlabel('Combined Artifact Score', fontsize=11)
ax.set_ylabel('Number of Epochs', fontsize=11)
ax.set_title('Artifact Score Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Score components over time
ax = axes[0, 1]
epoch_nums = np.arange(len(artifact_scores))
ax.scatter(epoch_nums[good_indices], artifact_scores[good_indices], 
           color='g', alpha=0.6, s=30, label=f'Good (n={len(good_indices)})')
ax.scatter(epoch_nums[artifact_indices], artifact_scores[artifact_indices], 
           color='r', alpha=0.6, s=30, label=f'Artifact (n={len(artifact_indices)})')
ax.axhline(details['threshold'], color='r', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch Index', fontsize=11)
ax.set_ylabel('Combined Artifact Score', fontsize=11)
ax.set_title('Artifact Score vs Epoch Number', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Individual score components
ax = axes[1, 0]
component_names = ['Spectral', 'Kurtosis', 'Zero-Cross', 'Variance']
component_good = [
    np.mean(details['spectral_scores'][good_indices]),
    np.mean(details['kurtosis_scores'][good_indices]),
    np.mean(details['zero_cross_scores'][good_indices]),
    np.mean(details['variance_scores'][good_indices]),
]
component_artifact = [
    np.mean(details['spectral_scores'][artifact_indices]),
    np.mean(details['kurtosis_scores'][artifact_indices]),
    np.mean(details['zero_cross_scores'][artifact_indices]),
    np.mean(details['variance_scores'][artifact_indices]),
]

x_pos = np.arange(len(component_names))
width = 0.35
ax.bar(x_pos - width/2, component_good, width, label='Good', color='g', alpha=0.7)
ax.bar(x_pos + width/2, component_artifact, width, label='Artifact', color='r', alpha=0.7)
ax.set_ylabel('Mean Score', fontsize=11)
ax.set_title('Average Scores by Component', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(component_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Box plots of individual components
ax = axes[1, 1]
data_to_plot = [
    details['spectral_scores'][good_indices],
    details['spectral_scores'][artifact_indices],
    details['kurtosis_scores'][good_indices],
    details['kurtosis_scores'][artifact_indices],
]
bp = ax.boxplot(data_to_plot, labels=['Spectral\n(Good)', 'Spectral\n(Artifact)', 
                                       'Kurtosis\n(Good)', 'Kurtosis\n(Artifact)'],
                patch_artist=True)
for i, box in enumerate(bp['boxes']):
    box.set_facecolor('g' if i % 2 == 0 else 'r')
    box.set_alpha(0.7)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Score Distributions (Good vs Artifact)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 2: COMPARE GOOD AND ARTIFACT EPOCHS
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
n_ch = n_channels

# Find most representative good and artifact epochs
if len(good_indices) > 0:
    good_epoch_idx = good_indices[np.argmin(artifact_scores[good_indices])]
else:
    good_epoch_idx = 0

if len(artifact_indices) > 0:
    artifact_epoch_idx = artifact_indices[np.argmax(artifact_scores[artifact_indices])]
else:
    artifact_epoch_idx = 0

cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, n_ch))

# Plot good epoch
ax = axes[0]
for ch in range(n_ch):
    ax.plot(times * 1000, merged_edata_uV[good_epoch_idx, ch, :], 
            color=colors[ch], alpha=0.7, lw=1.2, label=ch_names[ch])
ax.set_title(f"Example GOOD Epoch (Index {good_epoch_idx}, Score: {artifact_scores[good_epoch_idx]:.3f})", 
             fontsize=12, fontweight='bold', color='green')
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
ax.set_ylabel('µV')
ax.legend(loc='upper right', fontsize=8, ncol=3)

# Plot artifact epoch
ax = axes[1]
for ch in range(n_ch):
    ax.plot(times * 1000, merged_edata_uV[artifact_epoch_idx, ch, :], 
            color=colors[ch], alpha=0.7, lw=1.2, label=ch_names[ch])
ax.set_title(f"Example ARTIFACT Epoch (Index {artifact_epoch_idx}, Score: {artifact_scores[artifact_epoch_idx]:.3f})", 
             fontsize=12, fontweight='bold', color='red')
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
ax.set_ylabel('µV')
ax.set_xlabel('Time (ms)')
ax.legend(loc='upper right', fontsize=8, ncol=3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 3: SPECTRAL COMPARISON
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

freq_range = (0, 100)  # Hz

for idx, (ax, epoch_idx, label, color) in enumerate([
    (axes[0], good_epoch_idx, 'GOOD', 'green'),
    (axes[1], artifact_epoch_idx, 'ARTIFACT', 'red')
]):
    
    # Average power spectrum across channels
    power_spectra = []
    for ch in range(n_ch):
        signal_ch = merged_edata_uV[epoch_idx, ch, :]
        freqs = np.fft.fftfreq(len(signal_ch), 1/sfreq)
        fft_vals = np.fft.fft(signal_ch)
        power = np.abs(fft_vals) ** 2
        
        # Keep only positive frequencies
        pos_idx = freqs > 0
        freqs_pos = freqs[pos_idx]
        power_pos = power[pos_idx]
        
        power_spectra.append(power_pos)
    
    # Average across channels
    avg_power = np.mean(power_spectra, axis=0)
    freqs_pos_mean = freqs[freqs > 0][:len(avg_power)]
    
    ax.semilogy(freqs_pos_mean, avg_power, color=color, linewidth=2)
    ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (µV²)', fontsize=11)
    ax.set_title(f'{label} Epoch - Power Spectrum\n(Score: {artifact_scores[epoch_idx]:.3f})', 
                 fontsize=12, fontweight='bold', color=color)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# ============================================================================
# CREATE CLEANED DATASET
# ============================================================================

print("\n" + "=" * 80)
print("CREATING CLEANED DATASET")
print("=" * 80)

print(f"\nOriginal data shape: {merged_edata_uV.shape}")
# Filter epochs
merged_edata_uV = merged_edata_uV[good_indices]
merged_edata = merged_edata_uV/1e6
merged_epochs = merged_epochs[good_indices]
n_epochs_merged, n_channels, n_times = merged_edata_uV.shape

print(f"Cleaned data shape: {merged_edata_uV.shape}")
print(f"Removed {len(artifact_indices)} epochs ({details['percent_artifact']:.1f}%)")


# ============================================================================
# VISUALIZATION 1: PER-CHANNEL MEAN ACROSS ALL MERGED EPOCHS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Calculate per-channel mean across all epochs
mean_per_channel = np.nanmean(merged_edata_uV, axis=0)  # (n_channels, n_times)

# Plot 1: Mean signal per channel
n_cols = 6
n_rows = int(math.ceil(n_channels / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

for i in range(n_channels):
    ax = axes[i]
    ax.plot(times, mean_per_channel[i], color='C0', lw=1.5)
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
    ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.set_title(ch_names[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, alpha=0.3)

for ax in axes[n_channels:]:
    ax.axis('off')

fig.suptitle(f"Merged Data: Per-channel mean across all {n_epochs_merged} epochs (µV)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ============================================================================
# VISUALIZATION 2: ALL EPOCHS PER CHANNEL (OVERLAY WITH MEAN)
# ============================================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

for ch in range(n_channels):
    ax = axes[ch]
    
    # Plot all epochs for this channel (semi-transparent)
    for ep in range(n_epochs_merged):
        ax.plot(times, merged_edata_uV[ep, ch, :], color='C0', alpha=0.15, lw=0.5)
    
    # Plot mean on top
    mean_trace = np.nanmean(merged_edata_uV[:, ch, :], axis=0)
    ax.plot(times, mean_trace, color='C1', lw=2.0, label='Mean')
    
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
    ax.set_title(ch_names[ch], fontsize=8)
    if ch % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, alpha=0.25)
    if ch == 0:
        ax.legend(loc='upper right', fontsize=8)

for ax in axes[n_channels:]:
    ax.axis('off')

fig.suptitle(f"Merged Epochs: All {n_epochs_merged} trials per channel (overlay) with mean", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ============================================================================
# VISUALIZATION 3: EVOKED RESPONSE (AVERAGE ACROSS ALL EPOCHS)
# ============================================================================

evoked = merged_epochs.average()
data_uV = evoked.data * 1e6
n_ch = len(ch_names)

cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, n_ch))

fig = plt.figure(figsize=(11, 5))
ax = fig.add_axes([0.06, 0.12, 0.72, 0.82])

lines = []
for i in range(n_ch):
    ln, = ax.plot(times, data_uV[i], color=colors[i], lw=1.2, label=ch_names[i])
    lines.append(ln)

ax.set_xlim(times[0], times[-1])
ax.axvline(0.0, color='k', linestyle='--', alpha=0.6, label='Stimulus')
ax.axhline(0.0, color='k', linestyle='-', alpha=0.4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('µV')
ax.set_title(f'Merged EEG ({n_ch} channels) — mean across {n_epochs_merged} epochs')
ax.grid(True, alpha=0.3)

# Inset with channel layout
inset_ax = fig.add_axes([0.80, 0.62, 0.18, 0.30])
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_aspect('equal')

try:
    layout = mne.find_layout(evoked.info)
    pos = np.asarray(layout.pos)
    layout_names = list(layout.names)
    name_to_idx = {name: idx for idx, name in enumerate(layout_names)}

    for i, ch in enumerate(ch_names):
        idx = name_to_idx.get(ch, name_to_idx.get(ch.replace('EEG ', '').strip(), None))
        if idx is None:
            continue
        
        coords = np.asarray(pos[idx])
        x, y = float(coords[0]), float(coords[1])
        
        inset_ax.scatter(x, y, color=colors[i], edgecolor='k', s=30, zorder=11)
        label = ch.replace('EEG ', '').strip()
        inset_ax.text(x, y + 0.025, label, color=colors[i], fontsize=8,
                      ha='center', va='bottom', zorder=12,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

    inset_ax.set_title('Channels', fontsize=9)
    inset_ax.autoscale(False)
except Exception as e:
    print(f"Warning: Could not create layout inset: {e}")
    inset_ax.text(0.5, 0.5, 'Layout unavailable', ha='center', va='center', transform=inset_ax.transAxes)

plt.show()

# ============================================================================
# VISUALIZATION 4: TOPOMAPS
# ============================================================================

try:
    print("Creating topomaps...")
    times_top = np.arange(0.0, epoch_tmax, 0.01)  # Every 10ms
    evoked.plot_topomap(times_top,  vlim=(-100, 100))
    plt.show()
except Exception as e:
    print(f"Warning: Could not create topomaps: {e}")



print("\nProcessing complete!")

# ========================== Epoch correlation across channels ======================

def compute_epoch_correlation_timeseries(edata, times):
    """
    For each time point, compute how correlated all epochs are across channels.
    High correlation = artifact (consistent, stereotyped response).
    Low correlation = variable neurophysiological activity.
    """
    n_epochs, n_channels, n_times = edata.shape
    
    # Correlation over time
    correlation_over_time = np.zeros(n_times)
    
    for t in range(n_times):
        # Get all channels at this time point across all epochs
        data_at_t = edata[:, :, t]  # shape: (n_epochs, n_channels)
        
        # Flatten and compute correlation matrix
        flat_data = data_at_t.reshape(n_epochs, -1)
        
        # Mean absolute correlation between all samples at this timepoint
        corr_matrix = np.corrcoef(flat_data.T)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        correlation_over_time[t] = np.nanmean(np.abs(upper_triangle))
    
    return correlation_over_time

# Usage:
corr_time = compute_epoch_correlation_timeseries(merged_edata, times)

plt.figure(figsize=(12, 4))
plt.plot(times, corr_time, linewidth=2)
plt.fill_between(times, corr_time, alpha=0.3)
plt.xlabel('Time after stimulus (s)')
plt.ylabel('Mean Absolute Correlation')
plt.title('Signal Similarity Over Time (High = Artifact/Stereotyped Response)')
plt.axvline(0, color='r', linestyle='--', label='Stimulus onset')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Identify high-coherence window (artifact)
threshold = np.percentile(corr_time, 90)
artifact_times = times[corr_time > threshold]
print(f"Likely artifact period: {artifact_times[0]:.3f}s to {artifact_times[-1]:.3f}s")


# # # =========================== Set bipolar montage ================================
# # helpers_bipolar_plot.py
# from typing import List, Tuple, Optional
# import numpy as np
# import math
# import matplotlib.pyplot as plt

# # ---------- Create bipolar from numpy array ----------
# def create_bipolar_from_array(
#     edata: np.ndarray,
#     ch_names: List[str],
#     bipolar_pairs: List[Tuple[str, str, str]],
#     skip_missing: bool = True
# ) -> Tuple[np.ndarray, List[str]]:
#     """
#     edata: either (n_epochs, n_ch, n_times) or (n_ch, n_times)
#     ch_names: list of original channel names (length n_ch)
#     bipolar_pairs: list of tuples (anode_name, cathode_name, new_name)
#     returns:
#       edata_bip: (n_epochs, n_bip, n_times) if input has epochs, else (n_bip, n_times)
#       bip_names: list of new_name for included pairs
#     """
#     ch_to_idx = {name: i for i, name in enumerate(ch_names)}
#     # Determine input shape
#     if edata.ndim == 3:
#         n_epochs, _, n_times = edata.shape
#         has_epochs = True
#     elif edata.ndim == 2:
#         _, n_times = edata.shape
#         has_epochs = False
#     else:
#         raise ValueError("edata must be 2D (ch x times) or 3D (epochs x ch x times)")

#     bip_list = []
#     bip_names = []
#     for anode, cathode, new_name in bipolar_pairs:
#         if (anode in ch_to_idx) and (cathode in ch_to_idx):
#             ia = ch_to_idx[anode]
#             ic = ch_to_idx[cathode]
#             if has_epochs:
#                 bip = edata[:, ia, :] - edata[:, ic, :]  # shape (n_epochs, n_times)
#             else:
#                 bip = edata[ia, :] - edata[ic, :]        # shape (n_times,)
#             bip_list.append(bip)
#             bip_names.append(new_name)
#         else:
#             msg = f"Missing channel for pair ({anode}, {cathode})"
#             if skip_missing:
#                 print("Skipping pair because channel not found:", msg)
#                 continue
#             else:
#                 raise KeyError(msg)

#     if not bip_list:
#         raise RuntimeError("No bipolar pairs were created (check channel names or pairs).")

#     # stack into final array
#     if has_epochs:
#         # bip_list is list of (n_epochs, n_times) -> stack axis=1 -> (n_epochs, n_bip, n_times)
#         edata_bip = np.stack(bip_list, axis=1)
#     else:
#         # bip_list is list of (n_times,) -> stack -> (n_bip, n_times)
#         edata_bip = np.stack(bip_list, axis=0)

#     return edata_bip, bip_names


# # ---------- Create bipolar from MNE Epochs ----------
# def create_bipolar_from_epochs(
#     epochs,  # MNE Epochs
#     bipolar_pairs: List[Tuple[str, str, str]],
#     drop_refs: bool = True,
#     skip_missing: bool = True,
#     return_epochsarray: bool = False
# ):
#     """
#     epochs: mne.Epochs object
#     bipolar_pairs: list of (anode_name, cathode_name, new_name)
#     If return_epochsarray=True, attempts to create mne.EpochsArray of the bipolar signals (requires mne).
#     Returns: (edata_bip, bip_names) where edata_bip shape = (n_epochs, n_bip, n_times).
#     """
#     try:
#         import mne
#     except Exception:
#         raise ImportError("mne is required for create_bipolar_from_epochs")

#     raw_data = epochs.get_data()  # (n_epochs, n_ch, n_times)
#     ch_names = epochs.ch_names
#     edata_bip, bip_names = create_bipolar_from_array(raw_data, ch_names, bipolar_pairs, skip_missing=skip_missing)

#     if return_epochsarray:
#         # Build MNE info for bipolar channels (copy sampling frequency and set new ch_names)
#         sfreq = epochs.info['sfreq']
#         ch_types = ['eeg'] * len(bip_names)
#         info = mne.create_info(ch_names=bip_names, sfreq=sfreq, ch_types=ch_types)
#         # edata_bip is (n_epochs, n_bip, n_times)
#         epochs_bip = mne.EpochsArray(edata_bip, info, tmin=epochs.tmin, baseline=epochs.baseline)
#         return edata_bip, bip_names, epochs_bip

#     return edata_bip, bip_names


# # ---------- Plot bipolar averages ----------
# def plot_bipolar_averages(
#     edata_bip,
#     times_ms: np.ndarray,
#     bip_names: Optional[List[str]] = None,
#     ncols: Optional[int] = None,
#     figsize: Tuple[float, float] = (12, 6),
#     title: Optional[str] = None,
#     marker: str = '.',
#     marker_size: float = 6.0,
#     show: bool = True
# ):
#     """
#     Plot averages per bipolar pair.
#     edata_bip: either (n_epochs, n_bip, n_times) or (n_bip, n_times)
#     times_ms: vector (n_times,) in ms
#     bip_names: optional list of names length n_bip
#     """
#     times_ms = np.asarray(times_ms)
#     if edata_bip.ndim == 3:
#         mean_signals = edata_bip.mean(axis=0)  # (n_bip, n_times)
#     elif edata_bip.ndim == 2:
#         mean_signals = edata_bip
#     else:
#         raise ValueError("edata_bip must be 2D (n_bip x n_times) or 3D (n_epochs x n_bip x n_times)")

#     n_bip, n_times = mean_signals.shape
#     if times_ms.shape[0] != n_times:
#         raise ValueError("times_ms length must match number of samples in edata_bip")

#     if bip_names is None:
#         bip_names = [f"bip_{i}" for i in range(n_bip)]
#     if len(bip_names) != n_bip:
#         raise ValueError("bip_names length must match number of bipolar channels")

#     if ncols is None:
#         ncols = int(math.ceil(math.sqrt(n_bip)))
#     nrows = int(math.ceil(n_bip / ncols))

#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
#     axes_flat = axes.flatten()

#     for i in range(n_bip):
#         ax = axes_flat[i]
#         y = mean_signals[i, :]
#         ax.plot(times_ms, y, color='k', lw=1)
#         ax.set_title(bip_names[i], fontsize=9)
#         ax.set_xlabel("Time (ms)")
#         ax.set_ylabel("Amplitude (µV)")
#         ax.grid(True, lw=0.4, alpha=0.4)
#         ax.set_xlim(times_ms.min(), times_ms.max())

#     # hide unused axes
#     for j in range(n_bip, len(axes_flat)):
#         axes_flat[j].axis('off')

#     if title:
#         fig.suptitle(title, fontsize=12)
#         fig.tight_layout(rect=[0, 0, 1, 0.96])
#     else:
#         fig.tight_layout()

#     if show:
#         plt.show()

#     return fig, mean_signals


# # ----------------- Example usage (adapt to your chosen pairs) -----------------
# # Define the bipolar pairs you want (anode, cathode, new_name)
# bipolar_pairs = [
#     ('EEG F3', 'EEG Fz', 'F3-Fz'),
#     ('EEG Fz', 'EEG F4', 'Fz-F4'),
#     ('EEG F4', 'EEG F8', 'F4-F8'),
#     ('EEG T3', 'EEG C3', 'T3-C3'),
#     ('EEG C3', 'EEG Cz', 'C3-Cz'),
#     ('EEG Cz', 'EEG C4', 'Cz-C4'),
#     ('EEG T4', 'EEG T5', 'T4-T5'),
#     ('EEG P3', 'EEG Pz', 'P3-Pz'),
#     ('EEG Fp1', 'EEG F3', 'Fp1-F3'),
#     ('EEG F3', 'EEG T3', 'F3-T3'),
#     ('EEG T3', 'EEG T4', 'T3-T4'),
#     ('EEG Fpz', 'EEG T6', 'Fpz-T6'),
#     ('EEG T6', 'EEG Fz', 'T6-Fz'),
#     ('EEG Fz', 'EEG C3', 'Fz-C3'),
#     ('EEG C3', 'EEG T5', 'C3-T5'),
#     ('EEG Fp2', 'EEG F4', 'Fp2-F4'),
#     ('EEG F7', 'EEG Cz', 'F7-Cz'),
#     ('EEG Cz', 'EEG P3', 'Cz-P3'),
#     ('EEG F7', 'EEG F8', 'F7-F8'),
#     ('EEG F8', 'EEG C4', 'F8-C4'),
#     ('EEG C4', 'EEG Cz', 'C4-Cz')
# ]

# # bipolar_pairs = [('EEG F3', 'EEG Fz', 'F3-Fz'), ...] as you defined
# edata_bip, bip_names, epochs_bip = create_bipolar_from_epochs(epochs, bipolar_pairs, return_epochsarray=True)
# # times (ms)
# times_ms = epochs.times * 1000.0
# fig, means = plot_bipolar_averages(edata_bip*1000, times_ms, bip_names, ncols=3, figsize=(12,4), title="Bipolar averages")


# PSD
merged_epochs.average().compute_psd(
    method='welch', 
    fmin=0, 
    fmax=200).plot()


mne.viz.plot_epochs_image(
    merged_epochs,
    sigma=0.5,
    vmin=-250,
    vmax=250,
    show=True,
)


def _lighten_color(color, amount=0.6):
    """Return a lighter color by blending `color` with white.
    amount in [0,1]: 0 -> original color, 1 -> white"""
    try:
        r, g, b = to_rgb(color)
    except Exception:
        r, g, b = color
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return (r, g, b)


def plot_channel_means_with_envelope_same_color_ms(
    edata_sources: Optional[Dict[str, Any]] = None,
    times: Optional[np.ndarray] = None,
    ch_names: Optional[list] = None,
    sfreq: Optional[float] = None,
    n_cols: int = 6,
    mean_color='C1',
    mean_lw: float = 1.6,
    envelope_alpha: float = 0.35,      # slightly darker than before
    envelope_kind: str = 'std',        # 'std' or 'sem'
    envelope_lighten_frac: float = 0.50,  # less lightening => darker envelope
    vmax_abs_convert: float = 1e-2,
    figsize_per_col: tuple = (3, 2.2),
    return_fig_ax: bool = False
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Plot means +/- envelope for each channel, ALL subplots using the same color,
    show x axis time (ms) and make envelope a bit darker.

    See docstring in the previous version for parameter behavior and edata discovery.
    """
    # --- locate epoch data ---
    edata = None
    if edata_sources is None:
        candidates = ['edata', 'edata_uV', 'edata_good', 'edata_baselined',
                      'edata_cropped_all', 'epochs_data']
        g = globals()
        for name in candidates:
            if name in g and g[name] is not None:
                arr = g[name]
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    edata = arr
                    break
        if edata is None and 'epochs' in g:
            try:
                edata = g['epochs'].get_data()
            except Exception:
                edata = None
    else:
        for name, arr in edata_sources.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                edata = arr
                break

    if edata is None:
        raise RuntimeError("No epoch data found. Provide edata (n_epochs,n_ch,n_times) or an epochs object.")

    n_epochs, n_ch, n_times = edata.shape

    # --- times ---
    if times is None:
        g = globals()
        if 'times_cropped_all' in g:
            times = g['times_cropped_all']
        elif 'times' in g:
            times = g['times']
        elif 'epochs' in g:
            try:
                times = g['epochs'].times
            except Exception:
                times = None
        if times is None:
            if sfreq is None:
                if 'epochs' in g and hasattr(g['epochs'], 'info'):
                    sfreq = g['epochs'].info.get('sfreq', None)
            if sfreq is None:
                raise RuntimeError("No time vector available and sfreq not provided.")
            times = np.arange(n_times) / float(sfreq)

    times = np.asarray(times)
    # If times look like seconds (max reasonably small) convert to ms; if already >1000 assume ms
    if np.nanmax(np.abs(times)) <= 1000:
        times_ms = times * 1000.0
    else:
        times_ms = times.copy()

    # --- channel names ---
    if ch_names is None:
        g = globals()
        if 'epochs' in g:
            try:
                ch_names = list(g['epochs'].ch_names)
            except Exception:
                ch_names = None
        if ch_names is None:
            ch_names = [f"ch{i}" for i in range(n_ch)]
    assert len(ch_names) == n_ch, f"Channel name count ({len(ch_names)}) != data channels ({n_ch})"

    # --- units: convert to µV if necessary ---
    if np.nanmax(np.abs(edata)) < vmax_abs_convert:
        edata_uV = edata * 1e6
    else:
        edata_uV = edata.copy()

    # --- colors: same color for all channels; envelope is lighter (but darker than before) ---
    mean_color_rgb = to_rgb(mean_color)
    envelope_color = _lighten_color(mean_color_rgb, amount=envelope_lighten_frac)

    # --- figure & axes ---
    n_rows = int(math.ceil(n_ch / n_cols))
    figsize = (figsize_per_col[0] * n_cols, figsize_per_col[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
    axes = np.array(axes).reshape(-1)

    # Pre-compute xticks for ms axis (same ticks for all subplots)
    n_xticks = 4
    xticks = np.linspace(times_ms[0], times_ms[-1], n_xticks)
    xtick_labels = [f"{int(x)}" for x in xticks]

    for ch in range(n_ch):
        ax = axes[ch]

        # compute mean and envelope
        mean_trace = np.nanmean(edata_uV[:, ch, :], axis=0)
        std_trace = np.nanstd(edata_uV[:, ch, :], axis=0)
        if envelope_kind == 'sem':
            valid_counts = np.sum(~np.isnan(edata_uV[:, ch, :]), axis=0)
            sem = np.zeros_like(std_trace)
            mask = valid_counts > 0
            sem[mask] = std_trace[mask] / np.sqrt(valid_counts[mask])
            lower = mean_trace - sem
            upper = mean_trace + sem
        else:
            lower = mean_trace - std_trace
            upper = mean_trace + std_trace

        # plot shaded envelope (lightened, but darker than before) and mean line (solid)
        ax.fill_between(times_ms, lower, upper, color=envelope_color, alpha=envelope_alpha, linewidth=0)
        ax.plot(times_ms, mean_trace, color=mean_color_rgb, lw=mean_lw)

        ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
        ax.set_title(ch_names[ch], fontsize=8)

        # show x axis values (ms) on every subplot, as requested
        ax.set_xlabel('Time (ms)')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=8)

        if ch % n_cols == 0:
            ax.set_ylabel('µV')
        ax.set_xlim(times_ms[0], times_ms[-1])
        ax.grid(True, alpha=0.2)

    # turn off unused axes
    for ax in axes[n_ch:]:
        ax.axis('off')

    fig.suptitle("Channel means ± envelope (same color for all channels)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if return_fig_ax:
        return fig, axes
    else:
        plt.show()
        return None

# ============ Plot channel means with std envelope ========================

plot_channel_means_with_envelope_same_color_ms(
    edata_sources={'edata': merged_edata},
    times=times,
    ch_names=ch_names,
    mean_color='tab:blue',
    envelope_kind='std',
    envelope_alpha=0.35,
    envelope_lighten_frac=0.50,
    return_fig_ax=False
)


# ================== CROP STIMULUS ARTIFACT ==================
print("\n" + "="*50)
print("CROPPING STIMULUS ARTIFACT")
print("="*50)

# Define stimulus artifact duration
crop_duration_ms = 50 # should be 10.5
crop_duration_s = crop_duration_ms / 1000.0
crop_samples = int(np.round(crop_duration_s * sfreq))

print(f"Removing stimulus artifact: {crop_duration_ms} ms ({crop_samples} samples)")

# Remove the first N samples (stimulus artifact period)
# After removal: data starts at ~10.5 ms after stimulus onset
edata_cropped_all = merged_edata_uV[:, :, crop_samples:]
orig_times = times[crop_samples:]

# make cropped times start at 0.0 seconds
times_cropped_all = orig_times - orig_times[0]    # seconds, first sample = 0.0
# convenience ms vector (starts at 0 ms)
times_cropped_all_ms = times_cropped_all*1000

print(f"After cropping:")
print(f"  Data shape: {edata_cropped_all.shape}")
print(f"  Time range: [{times_cropped_all[0]:.4f}, {times_cropped_all[-1]:.4f}] s")


# ================== COMPUTE EVOKED POTENTIAL ==================
print("\n" + "="*50)
print("COMPUTING EVOKED POTENTIAL")
print("="*50)

# Average across good epochs
evoked_potential = np.mean(edata_cropped_all, axis=0)  # shape: (n_channels, n_times)

print(f"Evoked potential shape: {evoked_potential.shape}")

# ================== VISUALIZE RESULTS ==================
print("\n" + "="*50)
print("VISUALIZING RESULTS")
print("="*50)

# Get channel names for good epochs
eeg_labels = list(merged_epochs.ch_names)
# Plot evoked potential
n_ch = evoked_potential.shape[0]
n_cols = 6
n_rows = int(np.ceil(n_ch / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

for i in range(n_ch):
    ax = axes[i]
    ax.plot(times_cropped_all_ms, evoked_potential[i], color='C0', lw=1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset')
    ax.set_title(eeg_labels[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times_cropped_all_ms[0], times_cropped_all_ms[-1])
    ax.set_xlabel('Time (ms)')
    ax.grid(True, alpha=0.3)

for ax in axes[n_ch:]:
    ax.axis('off')

fig.suptitle(f"Evoked Potential After Stimulus Cropping\n({merged_edata.shape[0]} good epochs, stimulus artifact removed)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ================== PLOT COMPARISON ==================
print("\nPlotting comparison: before vs after cropping...")

# Show first channel as example
ch_idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# Before cropping
mean_before = np.mean(merged_edata[:, ch_idx, :]*1e6, axis=0)
ax1.plot(times, mean_before, color='C0', lw=1.5)
ax1.axvspan(0, crop_duration_s, alpha=0.3, color='red', label=f'Removed: {crop_duration_ms} ms')
ax1.axvline(0, color='k', linestyle='--', alpha=0.6)
ax1.set_ylabel('µV')
ax1.set_title(f'{eeg_labels[ch_idx]} - Before Cropping')
ax1.legend()
ax1.grid(True, alpha=0.3)

# After cropping
mean_after = np.mean(edata_cropped_all[:, ch_idx, :], axis=0)
ax2.plot(times_cropped_all*1000, mean_after, color='C1', lw=1.5)
ax2.axvline(0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset + artifact')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('µV')
ax2.set_title(f'{eeg_labels[ch_idx]} - After Cropping')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------- Plot cropped means per channel -------------------------

plot_channel_means_with_envelope_same_color_ms(
    edata_sources={'edata': edata_cropped_all},
    times=times_cropped_all,
    ch_names=ch_names,
    mean_color='tab:blue',
    envelope_kind='std',
    envelope_alpha=0.35,
    envelope_lighten_frac=0.50,
    return_fig_ax=False
)

# ====================== CCEP wave components ================================


"""
detect_and_plot_cceps_legend.py

Detect CCEP components and plot channel averages with a compact legend in the upper-right
corner of each subplot summarizing latencies and amplitudes for P1/N1/P2/N2.

Usage:
    - Import plot_channel_averages_with_peaks_legend and call it with mean_signals
      (n_channels x n_samples) and times_ms (length n_samples, in ms).
"""
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def detect_ccep_components_for_signal_on_mean(mean_signal: np.ndarray, times_ms: np.ndarray) -> Dict[str, Any]:
    """
    Detect P1, N1 (Matsumoto), P2, N2 components for a single mean trace.

    Returns a dict that includes:
      - latencies in ms
      - amplitudes (computed values)
      - sample indices (idx) relative to the mean_trace/time vector
      - y-values of mean_trace at those indices (so markers can be plotted exactly on the mean)
    """
    t = np.asarray(times_ms)
    y = np.asarray(mean_signal)

    # Define windows in ms (customize if needed)
    mask_section1 = (t >= 12) & (t <= 27)   # N1 search window
    mask_section2 = (t >= 12) & (t <= 50)   # P2 search window
    mask_section3 = (t >= 40) & (t <= 140)  # N2 search window
    mask_section4 = (t >= 6)  & (t <= 12)   # P1 search window

    res = {
        'P1_idx': None, 'P1_Latency': np.nan, 'P1_Amplitude': np.nan, 'P1_y': np.nan,
        'N1_idx': None, 'N1_Latency': np.nan, 'N1_Matsumoto': np.nan, 'N1_y': np.nan,
        'P2_idx': None, 'P2_Latency': np.nan, 'P2_Amplitude': np.nan, 'P2_y': np.nan,
        'N2_idx': None, 'N2_Latency': np.nan, 'N2_Amplitude': np.nan, 'N2_y': np.nan
    }

    def safe_min(mask: np.ndarray) -> Tuple[float, Optional[int]]:
        if mask.any():
            local = np.argmin(y[mask])
            global_idx = np.where(mask)[0][local]
            return float(y[global_idx]), int(global_idx)
        return np.nan, None

    def safe_max(mask: np.ndarray) -> Tuple[float, Optional[int]]:
        if mask.any():
            local = np.argmax(y[mask])
            global_idx = np.where(mask)[0][local]
            return float(y[global_idx]), int(global_idx)
        return np.nan, None

    min_val1, min_idx1 = safe_min(mask_section1)   # N1
    max_val2, max_idx2 = safe_max(mask_section2)   # P2
    min_val3, min_idx3 = safe_min(mask_section3)   # N2
    max_val4, max_idx4 = safe_max(mask_section4)   # P1

    # Fill P1
    if max_idx4 is not None:
        res['P1_idx'] = max_idx4
        res['P1_Latency'] = float(t[max_idx4])
        res['P1_Amplitude'] = float(max_val4)
        res['P1_y'] = float(y[max_idx4])

    # Fill N1
    if min_idx1 is not None:
        res['N1_idx'] = min_idx1
        res['N1_Latency'] = float(t[min_idx1])
        res['N1_y'] = float(y[min_idx1])

    # Fill P2
    if max_idx2 is not None:
        res['P2_idx'] = max_idx2
        res['P2_Latency'] = float(t[max_idx2])
        res['P2_Amplitude'] = float(max_val2)
        res['P2_y'] = float(y[max_idx2])

    # Fill N2
    if min_idx3 is not None:
        res['N2_idx'] = min_idx3
        res['N2_Latency'] = float(t[min_idx3])
        res['N2_y'] = float(y[min_idx3])
        if not np.isnan(max_val2) and not np.isnan(min_val3):
            res['N2_Amplitude'] = float(max_val2 - min_val3)

    # Compute Matsumoto N1 amplitude (intersection of P1-P2 line at x3 minus y3)
    try:
        if (max_idx4 is not None) and (max_idx2 is not None) and (min_idx1 is not None):
            x1, y1 = float(t[max_idx4]), float(max_val4)
            x2, y2 = float(t[max_idx2]), float(max_val2)
            x3, y3 = float(t[min_idx1]), float(min_val1)
            if not np.isclose(x2, x1):
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                y4 = m * x3 + c
                res['N1_Matsumoto'] = float(y4 - y3)
            else:
                res['N1_Matsumoto'] = np.nan
    except Exception:
        res['N1_Matsumoto'] = np.nan

    return res


def _format_val(val: Any, lat: bool = False) -> str:
    """Format value for legend label. Latencies: 1 decimal. Amplitudes: 2 decimal or 2 sig-fig for small values."""
    try:
        if val is None:
            return "-"
        fv = float(val)
        if np.isnan(fv):
            return "-"
        if lat:
            return f"{fv:.1f}"
        if abs(fv) >= 1:
            return f"{fv:.2f}"
        return f"{fv:.2g}"
    except Exception:
        return "-"


def plot_channel_averages_with_peaks_legend(
    mean_signals: np.ndarray,
    times_ms: np.ndarray,
    channel_names: Optional[List[str]] = None,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (14, 8),
    suptitle: Optional[str] = None,
    show_grid: bool = True,
    include_matsumoto: bool = True,
    marker_size: float = 6.0,
    legend_fontsize: float = 7.0,
    savepath: Optional[str] = None,
    show: bool = True
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Plot mean_signals (shape: n_channels x n_timepoints) as subplots and show a compact legend
    in the upper-right corner of each subplot listing latencies and amplitudes for P1/N1/P2/N2.

    Returns (detections_list, detections_dataframe).
    """
    mean_signals = np.asarray(mean_signals)
    times_ms = np.asarray(times_ms)
    if mean_signals.ndim != 2:
        raise ValueError("mean_signals must be 2D array (n_channels x n_samples)")
    n_ch, n_t = mean_signals.shape
    if times_ms.shape[0] != n_t:
        raise ValueError("times_ms length must match number of columns in mean_signals")

    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(n_ch)]
    if len(channel_names) != n_ch:
        raise ValueError("channel_names length must match number of channels")

    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n_ch)))
    nrows = int(math.ceil(n_ch / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    detections: List[Dict[str, Any]] = []

    for ch in range(n_ch):
        ax = axes_flat[ch]
        y = mean_signals[ch, :]
        det = detect_ccep_components_for_signal_on_mean(y, times_ms)
        detections.append(det)

        ax.plot(times_ms, y, color='k', lw=1)
        if show_grid:
            ax.grid(True, lw=0.5, alpha=0.3)

        ax.set_title(channel_names[ch], fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")

        # Markers to show peak positions
        def mark(idx_key: str, marker: str, color: str):
            idx = det.get(f"{idx_key}_idx")
            if idx is not None:
                x = times_ms[idx]
                yv = y[idx]
                ax.plot(x, yv, marker='o', color=color, markersize=marker_size, markeredgecolor='w')

        mark("P1", marker="^", color="tab:red")
        mark("N1", marker="v", color="tab:blue")
        mark("P2", marker="^", color="tab:orange")
        mark("N2", marker="v", color="tab:green")

        # Build compact legend entries
        # Always create four legend lines (P1,N1,P2,N2) with labels containing latency and amplitude (or "-" if missing)
        labels = []
        handles = []
        # definitions for each component
        comps = [
            ("P1", "^", "tab:red"),
            ("N1", "v", "tab:blue"),
            ("P2", "^", "tab:orange"),
            ("N2", "v", "tab:green"),
        ]
        for rl, mk, col in comps:
            lat = det.get(f"{rl}_Latency", np.nan)
            amp_val = det.get(f"{rl}_Amplitude", np.nan)
            if amp_val is None or (isinstance(amp_val, float) and np.isnan(amp_val)):
                amp_val = det.get(f"{rl}_y", np.nan)
            lat_str = _format_val(lat, lat=True)
            amp_str = _format_val(amp_val, lat=False)
            label = f"{rl}: {lat_str} ms, {amp_str} µV"
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markeredgecolor='k', markersize=marker_size, linestyle=''))
            labels.append(label)

        # Optionally append Matsumoto as a text-only legend entry
        if include_matsumoto:
            mats = det.get("N1_Matsumoto", np.nan)
            mats_str = _format_val(mats, lat=False)
            # Create a plain line handle for mats (no marker)
            handles.append(Line2D([0], [0], linestyle='-', color='none'))
            labels.append(f"Mats N1: {mats_str} µV")

        # Place legend in upper-right with small font and semi-transparent box
        # Using bbox_to_anchor to nudge slightly inside the axes frame
        ax.legend(handles, labels, loc='upper right', fontsize=legend_fontsize,
                  framealpha=0.7, borderpad=0.3, handlelength=0.5, handletextpad=0.6)

        ax.set_xlim(times_ms.min(), times_ms.max())

    # hide remaining axes if any
    for idx in range(n_ch, len(axes_flat)):
        axes_flat[idx].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if suptitle else None)

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')

    if show:
        plt.show()

    # Build tidy DataFrame of detections
    rows_df = []
    for ch_idx, det in enumerate(detections):
        row = {
            "channel_idx": ch_idx,
            "channel": channel_names[ch_idx],
            "P1_Latency_ms": det.get("P1_Latency", np.nan),
            "P1_Amplitude_uV": det.get("P1_Amplitude", det.get("P1_y", np.nan)),
            "N1_Latency_ms": det.get("N1_Latency", np.nan),
            "N1_Matsumoto_uV": det.get("N1_Matsumoto", np.nan),
            "P2_Latency_ms": det.get("P2_Latency", np.nan),
            "P2_Amplitude_uV": det.get("P2_Amplitude", det.get("P2_y", np.nan)),
            "N2_Latency_ms": det.get("N2_Latency", np.nan),
            "N2_Amplitude_uV": det.get("N2_Amplitude", det.get("N2_y", np.nan)),
        }
        rows_df.append(row)
    df = pd.DataFrame(rows_df)

    return detections, df


# assume edata_cropped_all created in your snippet:
# edata_cropped_all shape should be (n_epochs, n_channels, n_samples) or (n_channels, n_samples)
edata = edata_cropped_all  # name you used in your snippet
times_ms = times_cropped_all_ms  # ms, starts at 0.0 after your cropping/subtraction

# If edata is epochs x channels x samples:
if edata_cropped_all.ndim == 3:
    # average across epochs -> shape (n_channels, n_samples)
    mean_signals = edata_cropped_all.mean(axis=0)
elif edata_cropped_all.ndim == 2:
    # already channels x samples
    mean_signals = edata_cropped_all
else:
    raise ValueError("edata_cropped_all must be 2D (ch x samples) or 3D (epochs x ch x samples)")


dets, df = plot_channel_averages_with_peaks_legend(
    mean_signals,
    times_ms,
    channel_names=ch_names,
    ncols=4,
    figsize=(14, 10),
    include_matsumoto=True,
    show=True
)



# =================================== ITC =====================================
try:
    from tensorpac.utils import ITC
except Exception as exc:
    raise ImportError("tensorpac is required to run ITC. Install with `pip install tensorpac`") from exc

def compute_itc_tensorpac_all_channels(
    edata,             # shape (n_epochs, n_channels, n_times)
    sfreq,             # sampling frequency (Hz)
    f_pha=[8, 12],     # freq specification: either [fmin, fmax] or array of freqs
    dcomplex='wavelet',# 'wavelet' or 'hilbert'
    cycle=3,
    width=7,
    edges=None,
    n_jobs=1,
    verbose=False
):
    """
    Compute ITC using tensorpac.utils.ITC for all channels.
    Returns:
      itc_all: numpy array shaped either
         (n_channels, n_times) for single band f_pha=[fmin,fmax]
         or (n_channels, n_freqs, n_times) for multi-frequency f_pha=array_like
      example_itc_obj_list: list of ITC objects (one per channel) in case you want to plot with tensorpac later
    """
    edata = np.asarray(edata)
    if edata.ndim != 3:
        raise ValueError("edata must be shape (n_epochs, n_channels, n_times)")

    n_epochs, n_ch, n_times = edata.shape
    itc_objs = []
    itc_list = []

    for ch in range(n_ch):
        x = edata[:, ch, :]  # (n_epochs, n_times)
        # create ITC object (this computes values inside)
        itc_obj = ITC(x, sfreq, f_pha=f_pha, dcomplex=dcomplex,
                      cycle=cycle, width=width, edges=edges,
                      n_jobs=n_jobs, verbose=verbose)
        itc_objs.append(itc_obj)

        # Try to extract the numeric ITC results from the object.
        # The common attribute name is `itc` (freq x time or time).
        if hasattr(itc_obj, "itc"):
            vals = np.asarray(itc_obj.itc)
        # fallbacks if attribute naming changed across versions:
        elif hasattr(itc_obj, "power"):
            vals = np.asarray(itc_obj.power)
        elif hasattr(itc_obj, "data"):
            vals = np.asarray(itc_obj.data)
        elif hasattr(itc_obj, "complex"):
            # compute ITC from complex TFR: mean over trials of phase vectors
            c = np.asarray(itc_obj.complex)  # (n_epochs, n_freqs, n_times) or (n_epochs, n_times)
            phase = np.angle(c)
            vals = np.abs(np.mean(np.exp(1j * phase), axis=0))
        else:
            # last resort: call plot() which forces computation but we still need numeric values
            # raise informative error so user can inspect `dir(itc_obj)`
            raise RuntimeError(f"Cannot find numeric ITC inside ITC object for channel {ch}. "
                               f"Inspect attributes: {dir(itc_obj)}")

        # Normalize shapes:
        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals[np.newaxis, :]   # (1, n_times) -> single freq
        elif vals.ndim == 2:
            # (n_freqs, n_times) OK
            pass
        else:
            raise RuntimeError(f"Unexpected ITC array shape for channel {ch}: {vals.shape}")

        itc_list.append(vals)

    # stack -> (n_channels, n_freqs, n_times)
    itc_all = np.stack(itc_list, axis=0)
    # if single freq (n_freqs == 1) squeeze to (n_channels, n_times)
    if itc_all.shape[1] == 1:
        itc_all = itc_all[:, 0, :]

    return itc_all, itc_objs

# Example usage with your cropped epochs:
# edata_cropped_all is already created and is multiplied by 1e6 earlier (µV)
# times_cropped_all, sfreq and ch_names are already in scope
freqs = np.linspace(4, 40, 20)   # example TF frequencies
itc_tf, itc_objs = compute_itc_tensorpac_all_channels(
    edata=edata_cropped_all,    # (n_epochs, n_channels, n_times)
    sfreq=sfreq,
    f_pha=freqs,                # returns (n_channels, n_freqs, n_times)
    dcomplex='wavelet',         # use Morlet wavelets for TF
    width=7,
    edges=10,
    n_jobs=1,
    verbose=False
)

print("Computed ITC TF array shape:", itc_tf.shape)
# Quick plot: show ITC time-frequency of channel 0
ch_idx = 0
if itc_tf.ndim == 3:
    plt.figure(figsize=(8, 4))
    vmax = np.percentile(itc_tf[ch_idx], 98)
    plt.imshow(itc_tf[ch_idx], aspect='auto', origin='lower',
               extent=[times_cropped_all[0], times_cropped_all[-1], freqs[0], freqs[-1]],
               vmin=0, vmax=vmax, cmap='viridis')
    plt.colorbar(label='ITC')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'ITC (ch={ch_names[ch_idx]})')
    plt.axvline(0, color='k', ls='--')
    plt.show()
else:
    # single-band case: itc_tf shape (n_channels, n_times)
    plt.figure(figsize=(9, 3))
    plt.plot(times_cropped_all, itc_tf[ch_idx], label=f'ITC {ch_names[ch_idx]}')
    plt.xlabel('Time (s)')
    plt.ylabel('ITC (0-1)')
    plt.title(f'ITC (ch={ch_names[ch_idx]})')
    plt.axvline(0, color='k', ls='--')
    plt.ylim(0, 1.0)
    plt.show()



def average_itc_per_channel(itc, ch_names, freqs=None, times=None, save_csv_path=None, plot_bar=True):
    """
    Compute average ITC per channel and return a sorted DataFrame.

    Parameters
    ----------
    itc : np.ndarray
        ITC array. Either shape (n_channels, n_freqs, n_times) or (n_channels, n_times).
    ch_names : list[str]
        Channel names length n_channels.
    freqs : array_like or None
        If provided and itc is TF, used for axis labels (optional).
    times : array_like or None
        Optional times vector.
    save_csv_path : str or None
        If provided, save the table to this CSV path.
    plot_bar : bool
        If True, plot a horizontal bar chart of mean ITC sorted descending.

    Returns
    -------
    df_sorted : pandas.DataFrame
        Columns: ['rank', 'channel', 'mean_itc']
    """
    itc = np.asarray(itc)
    n_ch = len(ch_names)

    if itc.ndim == 3:
        # (n_channels, n_freqs, n_times) -> average across freq & time
        mean_per_ch = itc.mean(axis=(1,2))
    elif itc.ndim == 2:
        # (n_channels, n_times) -> average across time
        mean_per_ch = itc.mean(axis=1)
    else:
        raise ValueError(f"Unexpected ITC array shape: {itc.shape}")

    # Build DataFrame
    df = pd.DataFrame({
        'channel': ch_names,
        'mean_itc': mean_per_ch
    })
    df['rank'] = df['mean_itc'].rank(ascending=False, method='min').astype(int)
    df_sorted = df.sort_values('mean_itc', ascending=False).reset_index(drop=True)
    # reorder columns
    df_sorted = df_sorted[['rank', 'channel', 'mean_itc']]

    # print nicely
    pd.set_option('display.float_format', lambda x: f"{x:.4f}")
    print("\nAverage ITC per channel (sorted):")
    print(df_sorted.to_string(index=False))

    # save CSV
    if save_csv_path:
        df_sorted.to_csv(save_csv_path, index=False)
        print(f"\nSaved table to: {save_csv_path}")

    # optional bar plot
    if plot_bar:
        plt.figure(figsize=(6, max(3, 0.25 * n_ch)))
        plt.barh(df_sorted['channel'][::-1], df_sorted['mean_itc'][::-1], color='C0')
        plt.xlabel('Mean ITC (0-1)')
        plt.title('Mean ITC per channel')
        plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.show()

    return df_sorted


# Example 1: If you computed TF ITC `itc_tf` (freqs array used when computing):
df_itc = average_itc_per_channel(itc_tf, ch_names, freqs=freqs, times=times_cropped_all,
                                 save_csv_path='itc_mean_per_channel.csv', plot_bar=True)

# Example 2: If you computed single-band ITC `itc_alpha`:
# df_itc = average_itc_per_channel(itc_alpha, ch_names,
#                                  save_csv_path='itc_mean_per_channel_alpha.csv', plot_bar=True)


# ============================================================================
# GROUP EPOCHS WITH SELECTABLE RANGE
# ============================================================================

# CONFIGURATION: Select epoch range and group size
epoch_range_start = 0          # First epoch to include
epoch_range_end = 100          # Last epoch to include (exclusive, so 0-100 = 100 epochs)
group_size = 10                # Number of epochs per group

# Validate range
n_total_epochs = edata_cropped_all.shape[0]
if epoch_range_end > n_total_epochs:
    epoch_range_end = n_total_epochs
    print(f"Warning: epoch_range_end exceeds total epochs. Set to {n_total_epochs}")

if epoch_range_start < 0:
    epoch_range_start = 0
    print(f"Warning: epoch_range_start is negative. Set to 0")

# Extract the selected epoch range
edata_selected = edata_cropped_all[epoch_range_start:epoch_range_end]

print("=" * 80)
print("GROUPING EPOCHS")
print("=" * 80)
print(f"Total epochs available: {n_total_epochs}")
print(f"Selected range: epochs {epoch_range_start} to {epoch_range_end-1} ({epoch_range_end - epoch_range_start} epochs)")
print(f"Group size: {group_size}\n")

n_selected_epochs = edata_selected.shape[0]
n_groups = int(np.floor(n_selected_epochs / group_size))
remainder = n_selected_epochs % group_size

print(f"Epochs in selected range: {n_selected_epochs}")
print(f"Number of complete groups: {n_groups}")
print(f"Remaining epochs: {remainder}\n")

# Create grouped averages
grouped_averages = []
group_info = []

for group_idx in range(n_groups):
    start_idx = group_idx * group_size
    end_idx = start_idx + group_size
    
    # Average this group
    group_data = edata_selected[start_idx:end_idx]
    group_mean = np.mean(group_data, axis=0)  # shape: (n_channels, n_times)
    
    grouped_averages.append(group_mean)
    
    # Store info with ABSOLUTE epoch indices (relative to full merged dataset)
    abs_start = epoch_range_start + start_idx
    abs_end = epoch_range_start + end_idx
    
    group_info.append({
        'group_num': group_idx + 1,
        'start_epoch_abs': abs_start,
        'end_epoch_abs': abs_end,
        'start_epoch_rel': start_idx,
        'end_epoch_rel': end_idx,
        'n_epochs': group_size
    })
    
    print(f"  Group {group_idx + 1}: epochs {abs_start}-{abs_end-1} (relative: {start_idx}-{end_idx-1})")

# Handle remainder
if remainder > 0:
    start_idx = n_groups * group_size
    end_idx = start_idx + remainder
    
    group_data = edata_selected[start_idx:end_idx]
    group_mean = np.mean(group_data, axis=0)
    
    grouped_averages.append(group_mean)
    
    abs_start = epoch_range_start + start_idx
    abs_end = epoch_range_start + end_idx
    
    group_info.append({
        'group_num': n_groups + 1,
        'start_epoch_abs': abs_start,
        'end_epoch_abs': abs_end,
        'start_epoch_rel': start_idx,
        'end_epoch_rel': end_idx,
        'n_epochs': remainder
    })
    
    print(f"  Group {n_groups + 1}: epochs {abs_start}-{abs_end-1} (remainder, n={remainder})")

# Convert to numpy array
grouped_averages = np.array(grouped_averages)  # shape: (n_groups, n_channels, n_times)

print(f"\nGrouped averages shape: {grouped_averages.shape}")
print(f"  (n_groups, n_channels, n_times) = ({grouped_averages.shape[0]}, {grouped_averages.shape[1]}, {grouped_averages.shape[2]})")

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
        ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
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
    title = (f"Group {info['group_num']}: Epochs {info['start_epoch_abs']}-{info['end_epoch_abs']-1} "
             f"(n={info['n_epochs']})")
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

print(f"\n✓ Generated {len(grouped_averages)} plots")

# ================== PLOT ALL GROUPS ON SAME FIGURE ==================
# Ensure n_rows and n_cols are defined
n_cols = 6  # Adjust as needed
n_rows = int(np.ceil(n_channels / n_cols))

# Create a figure showing all groups overlaid for each channel
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)

# Handle edge case where matplotlib returns 1D array for single row/column
if n_rows == 1 or n_cols == 1:
    axes = axes.reshape(n_rows, n_cols)

axes = axes.ravel()

# Define colors for different groups
colors = plt.cm.tab20(np.linspace(0, 1, len(grouped_averages)))

for ch_idx in range(n_channels):
    ax = axes[ch_idx]
    
    # Plot each group
    for group_idx, group_mean in enumerate(grouped_averages):
        label = f"Group {group_idx + 1}"
        ax.plot(times_cropped_all*1000, group_mean[ch_idx], 
                color=colors[group_idx], lw=1.5, label=label, alpha=0.7)
    
    ax.axvline(9.0, color='k', linestyle='--', alpha=0.3, label='P1')
    ax.axvline(20.0, color='k', linestyle='--', alpha=0.3, label='N1')
    ax.axvline(37.0, color='k', linestyle='--', alpha=0.3, label='P2')
    ax.axvline(89.0, color='k', linestyle='--', alpha=0.3, label='N2')
    ax.set_title(eeg_labels[ch_idx], fontsize=8)
    if ch_idx % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times_cropped_all[0]*1000, times_cropped_all[-1]*1000)
    ax.set_xlabel('ms')
    ax.grid(True, alpha=0.3)

# Hide extra subplots
for ax_idx in range(n_channels, len(axes)):
    axes[ax_idx].axis('off')

# Add legend (only show group labels to avoid duplication)
handles, labels = axes[0].get_legend_handles_labels()
# Filter to only include group labels (not the phase labels)
group_handles = handles[:len(grouped_averages)]
group_labels = labels[:len(grouped_averages)]
fig.legend(group_handles, group_labels, loc='upper center', 
           bbox_to_anchor=(0.5, -0.01), ncol=len(grouped_averages))

fig.suptitle("All Groups Overlay - Per Channel Average", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()


"""
detect_groups_and_plot.py

Apply peak detection (P1, N1, P2, N2 + Matsumoto N1) to grouped epoch averages,
plot each group's bipolar/channel averages and mark detected peaks with a point.
Also return/save a tidy DataFrame with detections for every group x channel x component.

Usage:
 - Place this file in the same namespace as your variables (grouped_averages, times_cropped_all_ms, ch_names, group_info)
 - Or import the functions and call them with those variables.

Functions:
 - detect_ccep_components_for_signal_on_mean(...)  # your detector (copied)
 - apply_detection_to_grouped_averages(grouped_averages, times_ms, channel_names)
 - plot_grouped_averages_with_detection(...)

Example (after you already computed grouped_averages and group_info as in your snippet):
    times_ms = times_cropped_all_ms  # ensure this exists and is in ms
    detections, df = apply_detection_to_grouped_averages(grouped_averages, times_ms, ch_names)
    plot_grouped_averages_with_detection(grouped_averages, times_ms, ch_names, group_info=group_info,
                                         detections=detections, ncols=6)
"""


# ---------------------  ---------------------
def detect_ccep_components_for_signal_on_mean(mean_signal: np.ndarray, times_ms: np.ndarray) -> Dict[str, Any]:
    t = np.asarray(times_ms)
    y = np.asarray(mean_signal)

    # Define windows in ms (customize if needed)
    mask_section1 = (t >= 12) & (t <= 27)   # N1 search window
    mask_section2 = (t >= 12) & (t <= 50)   # P2 search window
    mask_section3 = (t >= 40) & (t <= 140)  # N2 search window
    mask_section4 = (t >= 6)  & (t <= 12)   # P1 search window

    res = {
        'P1_idx': None, 'P1_Latency': np.nan, 'P1_Amplitude': np.nan, 'P1_y': np.nan,
        'N1_idx': None, 'N1_Latency': np.nan, 'N1_Matsumoto': np.nan, 'N1_y': np.nan,
        'P2_idx': None, 'P2_Latency': np.nan, 'P2_Amplitude': np.nan, 'P2_y': np.nan,
        'N2_idx': None, 'N2_Latency': np.nan, 'N2_Amplitude': np.nan, 'N2_y': np.nan
    }

    def safe_min(mask: np.ndarray) -> Tuple[float, Optional[int]]:
        if mask.any():
            local = np.argmin(y[mask])
            global_idx = np.where(mask)[0][local]
            return float(y[global_idx]), int(global_idx)
        return np.nan, None

    def safe_max(mask: np.ndarray) -> Tuple[float, Optional[int]]:
        if mask.any():
            local = np.argmax(y[mask])
            global_idx = np.where(mask)[0][local]
            return float(y[global_idx]), int(global_idx)
        return np.nan, None

    min_val1, min_idx1 = safe_min(mask_section1)   # N1
    max_val2, max_idx2 = safe_max(mask_section2)   # P2
    min_val3, min_idx3 = safe_min(mask_section3)   # N2
    max_val4, max_idx4 = safe_max(mask_section4)   # P1

    # Fill P1
    if max_idx4 is not None:
        res['P1_idx'] = max_idx4
        res['P1_Latency'] = float(t[max_idx4])
        res['P1_Amplitude'] = float(max_val4)
        res['P1_y'] = float(y[max_idx4])

    # Fill N1
    if min_idx1 is not None:
        res['N1_idx'] = min_idx1
        res['N1_Latency'] = float(t[min_idx1])
        res['N1_y'] = float(y[min_idx1])

    # Fill P2
    if max_idx2 is not None:
        res['P2_idx'] = max_idx2
        res['P2_Latency'] = float(t[max_idx2])
        res['P2_Amplitude'] = float(max_val2)
        res['P2_y'] = float(y[max_idx2])

    # Fill N2
    if min_idx3 is not None:
        res['N2_idx'] = min_idx3
        res['N2_Latency'] = float(t[min_idx3])
        res['N2_y'] = float(y[min_idx3])
        if not np.isnan(max_val2) and not np.isnan(min_val3):
            res['N2_Amplitude'] = float(max_val2 - min_val3)

    # Compute Matsumoto N1 amplitude (intersection of P1-P2 line at x3 minus y3)
    try:
        if (max_idx4 is not None) and (max_idx2 is not None) and (min_idx1 is not None):
            x1, y1 = float(t[max_idx4]), float(max_val4)
            x2, y2 = float(t[max_idx2]), float(max_val2)
            x3, y3 = float(t[min_idx1]), float(min_val1)
            if not np.isclose(x2, x1):
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                y4 = m * x3 + c
                res['N1_Matsumoto'] = float(y4 - y3)
            else:
                res['N1_Matsumoto'] = np.nan
    except Exception:
        res['N1_Matsumoto'] = np.nan

    return res

# --------------------- Apply detection to grouped averages ---------------------
def apply_detection_to_grouped_averages(
    grouped_averages: np.ndarray,
    times_ms: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> Tuple[List[List[Dict[str, Any]]], pd.DataFrame]:
    """
    Apply detect_ccep_components_for_signal_on_mean to each group average.
    grouped_averages: shape (n_groups, n_channels, n_times)
    times_ms: vector in ms length n_times
    channel_names: optional list of length n_channels

    Returns:
      - detections: nested list [group_idx][channel_idx] -> detection dict
      - df: tidy DataFrame with rows for group, channel, component, latency_ms, amplitude_uV
    """
    ga = np.asarray(grouped_averages)
    if ga.ndim != 3:
        raise ValueError("grouped_averages must be 3D array (n_groups, n_channels, n_times)")

    n_groups, n_channels, n_times = ga.shape
    if len(times_ms) != n_times:
        raise ValueError("times_ms length must match grouped_averages time dimension")

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError("channel_names length must match number of channels in grouped_averages")

    detections: List[List[Dict[str, Any]]] = []
    rows = []
    components = ["P1", "N1", "P2", "N2"]

    for g in range(n_groups):
        group_det = []
        for ch in range(n_channels):
            mean_trace = ga[g, ch, :]
            det = detect_ccep_components_for_signal_on_mean(mean_trace, times_ms)
            group_det.append(det)

            # add rows for P1/N1/P2/N2
            for comp in components:
                lat = det.get(f"{comp}_Latency", np.nan)
                amp = det.get(f"{comp}_Amplitude", det.get(f"{comp}_y", np.nan))
                rows.append({
                    "group": g + 1,
                    "channel_idx": ch,
                    "channel": channel_names[ch],
                    "component": comp,
                    "latency_ms": lat,
                    "amplitude_uV": amp
                })

            # add Matsumoto N1 row
            rows.append({
                "group": g + 1,
                "channel_idx": ch,
                "channel": channel_names[ch],
                "component": "N1_Matsumoto",
                "latency_ms": np.nan,
                "amplitude_uV": det.get("N1_Matsumoto", np.nan)
            })

        detections.append(group_det)

    df = pd.DataFrame(rows)
    return detections, df

# --------------------- Plot grouped averages with detection ---------------------
def plot_grouped_averages_with_detection(
    grouped_averages: np.ndarray,
    times_ms: np.ndarray,
    channel_names: Optional[List[str]] = None,
    group_info: Optional[List[Dict[str, Any]]] = None,
    detections: Optional[List[List[Dict[str, Any]]]] = None,
    ncols: int = 6,
    figsize_per_col: Tuple[float, float] = (3.0, 2.2),
    marker: str = 'o',
    marker_size: float = 4.0,
    legend_fontsize: float = 6.5,
    include_matsumoto: bool = True,
    save_prefix: Optional[str] = None,
    show: bool = True
) -> None:
    """
    For each group, plot channels in a grid, mark detected peaks and add compact legend per subplot.

    grouped_averages: (n_groups, n_channels, n_times)
    times_ms: times in ms
    channel_names: list of channel labels
    group_info: optional list with dicts for each group (e.g., start_epoch, end_epoch, n_epochs)
    detections: optional precomputed nested list [group][channel] detection dicts; if None, will compute
    save_prefix: if provided, will save png files named {save_prefix}_group_{g+1}.png
    """
    ga = np.asarray(grouped_averages)
    n_groups, n_channels, n_times = ga.shape
    times_ms = np.asarray(times_ms)
    if times_ms.shape[0] != n_times:
        raise ValueError("times_ms length must match grouped_averages time dimension")

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError("channel_names length must match number of channels")

    if detections is None:
        detections, _ = apply_detection_to_grouped_averages(ga, times_ms, channel_names)

    ncols = int(ncols)
    nrows = int(math.ceil(n_channels / ncols))
    fig_w = figsize_per_col[0] * ncols
    fig_h = figsize_per_col[1] * nrows

    for g in range(n_groups):
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        axes_flat = axes.flatten()

        for ch in range(n_channels):
            ax = axes_flat[ch]
            trace = ga[g, ch, :]
            ax.plot(times_ms, trace, color='C0', lw=1.2)
            # zero line and stimulus vertical at 0 ms
            ax.axhline(0, color='k', lw=0.6, alpha=0.5)
            ax.axvline(0.0, color='k', lw=0.8, linestyle='--', alpha=0.5)

            # mark detected peaks (as points)
            det = detections[g][ch]
            def mark_point(idx_key: str, color: str):
                idx = det.get(f"{idx_key}_idx")
                if idx is not None:
                    x = times_ms[idx]
                    yv = trace[idx]
                    ax.plot(x, yv, marker=marker, color=color, markersize=marker_size,
                            markeredgecolor='k', linestyle='')

            mark_point("P1", "tab:red")
            mark_point("N1", "tab:blue")
            mark_point("P2", "tab:orange")
            mark_point("N2", "tab:green")

            # compact legend text entries
            handles = []
            labels = []
            comps = [("P1","tab:red"), ("N1","tab:blue"), ("P2","tab:orange"), ("N2","tab:green")]
            for comp, col in comps:
                lat = det.get(f"{comp}_Latency", np.nan)
                amp = det.get(f"{comp}_Amplitude", det.get(f"{comp}_y", np.nan))
                lat_str = "-" if (lat is None or (isinstance(lat, float) and np.isnan(lat))) else f"{lat:.1f}"
                amp_str = "-" if (amp is None or (isinstance(amp, float) and np.isnan(amp))) else (f"{amp:.2f}" if abs(float(amp))>=1 else f"{float(amp):.2g}")
                labels.append(f"{comp}: {lat_str} ms, {amp_str} µV")
                handles.append(Line2D([0],[0], marker=marker, color='w', markerfacecolor=col,
                                      markeredgecolor='k', markersize=marker_size, linestyle=''))

            if include_matsumoto:
                mats = det.get("N1_Matsumoto", np.nan)
                mats_str = "-" if (isinstance(mats, float) and np.isnan(mats)) else (f"{mats:.2f}" if abs(float(mats))>=1 else f"{float(mats):.2g}")
                handles.append(Line2D([0],[0], linestyle='-', color='none'))
                labels.append(f"Mats N1: {mats_str} µV")

            # place legend in upper-right, small font
            ax.legend(handles, labels, loc='upper right', fontsize=legend_fontsize,
                      framealpha=0.7, borderpad=0.25, handlelength=0.6, handletextpad=0.5)

            ax.set_title(channel_names[ch], fontsize=8)
            if ch % ncols == 0:
                ax.set_ylabel('µV')
            ax.set_xlim(times_ms.min(), times_ms.max())
            ax.grid(True, alpha=0.25, lw=0.4)

        # hide extra axes
        for ax in axes_flat[n_channels:]:
            ax.axis('off')

        # title for group
        if group_info is not None and len(group_info) > g:
            info = group_info[g]
            title = (f"Group {info.get('group_num', g+1)}: Epochs {info.get('start_epoch','?')}-{info.get('end_epoch', '?')-1} (n={info.get('n_epochs','?')})")
        else:
            title = f"Group {g+1}"
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_prefix:
            fname = f"{save_prefix}_group_{g+1}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            print(f"Saved {fname}")

        if show:
            plt.show()
        else:
            plt.close(fig)

# --------------------- Example integration (adapt to your variables) ---------------------

# Apply detector to every group x channel
detections, df = apply_detection_to_grouped_averages(grouped_averages, times_cropped_all_ms, channel_names=ch_names)
print("Detections computed. Summary (first rows):")
print(df.head())


# Plot results, include legends and markers (will open one figure per group)
plot_grouped_averages_with_detection(grouped_averages, times_cropped_all_ms,
                                    channel_names=ch_names,
                                    group_info=group_info if 'group_info' in globals() else None,
                                    detections=detections,
                                    ncols=6,
                                    marker='o',
                                    marker_size=4.0,
                                    save_prefix="group_plot",
                                    show=True)

