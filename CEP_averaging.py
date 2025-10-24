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
file_path = r"C:\Users\msedo\Documents\Cerebelo\Chimula Mark\13.14_stim.edf"  # Change this to your file path

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
data = raw.get_data()  # shape (n_channels, n_samples)
labels = raw.ch_names
sfreq = float(raw.info['sfreq'])
signals = [data[i].astype(np.float64) for i in range(data.shape[0])]
sample_frequencies = [sfreq for _ in labels]
units = ['' for _ in labels]

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
filt_data = raw_filtered.get_data()[eeg_idx,:] * 3e6  # shape: (n_eeg, n_samples) in µV
filt_signals = [filt_data[i].astype(np.float64) for i in range(filt_data.shape[0])]

raw_data = raw.get_data()[eeg_idx,:]* 3e6
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
# ----------------------- Signal Visualisation ---------------------------------

# get events from annotations (if any)
events, event_id = mne.events_from_annotations(raw_filtered)  # events shape (n_events, 3)

# Launch interactive viewer with events displayed
# duration controls how many seconds are visible at once
raw_filtered.plot(n_channels=20, duration=30, scalings='auto', events=events, event_id=event_id)

#------------------------------------------------------------------------------


# --- create 4-second epochs starting at each event onset ---

# desired epoch length in seconds
epoch_tmin = 0.0
epoch_tmax = 4.0


print(f"Requested events: {len(events)}. Example events (first 10 rows):\n{events[:10]}")
print(f"Event ID mapping: {event_id}")

# Create epochs from raw_filtered (use filtered data in volts; MNE expects SI units)
raw_for_epochs = raw_filtered.copy()  # filtered in Volts

#raw_for_epochs.drop_channels(['EEG Fp1', 'EEG EOGI', 'EEG Fp2', 'EEG T3', 'EEG T4', 'EEG P4', 'EEG O1', 'EEG EOGD'])

# Create epochs: start at event (tmin=0.0) and last 4 seconds (tmax=4.0)
epochs = mne.Epochs(raw_for_epochs, events, event_id=event_id,
                    tmin=epoch_tmin, tmax=epoch_tmax,
                    picks=eeg_idx, baseline=None, preload=True, verbose=False)

n_requested = len(events)
n_kept = len(epochs.events)
n_dropped = n_requested - n_kept
print(f"Epochs requested: {n_requested}, kept: {n_kept}, dropped: {n_dropped}")

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
evoked.plot(spatial_colors=True, picks=eeg_idx)   # interactive MNE plot

# Or plot raw epochs (matplotlib)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(n_times) / sfreq, edata.mean(axis=0).T * 1e6)  # mean across epochs, in µV
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Mean epoch across events (µV)')
plt.show()




# PARAMETERS - tune as needed
search_window = (0.0, 0.25)    # seconds relative to original epoch onset in which to search for the first peak
epoch_duration_s = 4.0         # desired epoch length after alignment (seconds)
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
