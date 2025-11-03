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

# Specify the full path to your EDF file
file_path = r"C:\Users\msedo\Documents\CCEPs\Martin Garcia\CCEPs_13.10.edf"  # Change this to your file path

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

raw.drop_channels( ['Chin+','ECG+','DI+','DD+','RONQ+','CAN+','TER+','PCO2+','EtCO+','Pos+','Tor+','Abd+','TIBI+','TIBD+','thor+','abdo+','PULS+','BEAT+','SpO2+','MKR+'])
raw.drop_channels([
   'EEG Fp1', 'EEG Fp2', 'EEG EOGI', 'EEG T3', 'EEG T4',
   'EEG O1', 'EEG EOGD', 'EEG O2', 'EEG A1', 'EEG A2'
])

raw.drop_channels([
    'EEG P4', 'EEG T6'
])

# mapping: {old_name: new_name}
# mapping = {
#     'EEG F3': 'EEG F1',
#     'EEG F7': 'EEG F3',
#     'EEG C3' : 'EEG C1',
#     'EEG P3' : 'EEG CP1',
#     'EEG T5' : 'EEG CP3',
#     'EEG F4' : 'EEG F2',
#     'EEG F8' : 'EEG F4',
#     'EEG C4' : 'EEG C2',
#     'EEG P4' : 'EEG CP2',
#     'EEG T6' : 'EEG CP4'

# }

#raw.rename_channels(mapping)      # modifies raw in-place

print(raw.ch_names[:20])
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
epoch_tmax = 0.17

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


# # =========================== Set bipolar montage ================================
# helpers_bipolar_plot.py
from typing import List, Tuple, Optional
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------- Create bipolar from numpy array ----------
def create_bipolar_from_array(
    edata: np.ndarray,
    ch_names: List[str],
    bipolar_pairs: List[Tuple[str, str, str]],
    skip_missing: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    edata: either (n_epochs, n_ch, n_times) or (n_ch, n_times)
    ch_names: list of original channel names (length n_ch)
    bipolar_pairs: list of tuples (anode_name, cathode_name, new_name)
    returns:
      edata_bip: (n_epochs, n_bip, n_times) if input has epochs, else (n_bip, n_times)
      bip_names: list of new_name for included pairs
    """
    ch_to_idx = {name: i for i, name in enumerate(ch_names)}
    # Determine input shape
    if edata.ndim == 3:
        n_epochs, _, n_times = edata.shape
        has_epochs = True
    elif edata.ndim == 2:
        _, n_times = edata.shape
        has_epochs = False
    else:
        raise ValueError("edata must be 2D (ch x times) or 3D (epochs x ch x times)")

    bip_list = []
    bip_names = []
    for anode, cathode, new_name in bipolar_pairs:
        if (anode in ch_to_idx) and (cathode in ch_to_idx):
            ia = ch_to_idx[anode]
            ic = ch_to_idx[cathode]
            if has_epochs:
                bip = edata[:, ia, :] - edata[:, ic, :]  # shape (n_epochs, n_times)
            else:
                bip = edata[ia, :] - edata[ic, :]        # shape (n_times,)
            bip_list.append(bip)
            bip_names.append(new_name)
        else:
            msg = f"Missing channel for pair ({anode}, {cathode})"
            if skip_missing:
                print("Skipping pair because channel not found:", msg)
                continue
            else:
                raise KeyError(msg)

    if not bip_list:
        raise RuntimeError("No bipolar pairs were created (check channel names or pairs).")

    # stack into final array
    if has_epochs:
        # bip_list is list of (n_epochs, n_times) -> stack axis=1 -> (n_epochs, n_bip, n_times)
        edata_bip = np.stack(bip_list, axis=1)
    else:
        # bip_list is list of (n_times,) -> stack -> (n_bip, n_times)
        edata_bip = np.stack(bip_list, axis=0)

    return edata_bip, bip_names


# ---------- Create bipolar from MNE Epochs ----------
def create_bipolar_from_epochs(
    epochs,  # MNE Epochs
    bipolar_pairs: List[Tuple[str, str, str]],
    drop_refs: bool = True,
    skip_missing: bool = True,
    return_epochsarray: bool = False
):
    """
    epochs: mne.Epochs object
    bipolar_pairs: list of (anode_name, cathode_name, new_name)
    If return_epochsarray=True, attempts to create mne.EpochsArray of the bipolar signals (requires mne).
    Returns: (edata_bip, bip_names) where edata_bip shape = (n_epochs, n_bip, n_times).
    """
    try:
        import mne
    except Exception:
        raise ImportError("mne is required for create_bipolar_from_epochs")

    raw_data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    ch_names = epochs.ch_names
    edata_bip, bip_names = create_bipolar_from_array(raw_data, ch_names, bipolar_pairs, skip_missing=skip_missing)

    if return_epochsarray:
        # Build MNE info for bipolar channels (copy sampling frequency and set new ch_names)
        sfreq = epochs.info['sfreq']
        ch_types = ['eeg'] * len(bip_names)
        info = mne.create_info(ch_names=bip_names, sfreq=sfreq, ch_types=ch_types)
        # edata_bip is (n_epochs, n_bip, n_times)
        epochs_bip = mne.EpochsArray(edata_bip, info, tmin=epochs.tmin, baseline=epochs.baseline)
        return edata_bip, bip_names, epochs_bip

    return edata_bip, bip_names


# ---------- Plot bipolar averages ----------
def plot_bipolar_averages(
    edata_bip,
    times_ms: np.ndarray,
    bip_names: Optional[List[str]] = None,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    marker: str = '.',
    marker_size: float = 6.0,
    show: bool = True
):
    """
    Plot averages per bipolar pair.
    edata_bip: either (n_epochs, n_bip, n_times) or (n_bip, n_times)
    times_ms: vector (n_times,) in ms
    bip_names: optional list of names length n_bip
    """
    times_ms = np.asarray(times_ms)
    if edata_bip.ndim == 3:
        mean_signals = edata_bip.mean(axis=0)  # (n_bip, n_times)
    elif edata_bip.ndim == 2:
        mean_signals = edata_bip
    else:
        raise ValueError("edata_bip must be 2D (n_bip x n_times) or 3D (n_epochs x n_bip x n_times)")

    n_bip, n_times = mean_signals.shape
    if times_ms.shape[0] != n_times:
        raise ValueError("times_ms length must match number of samples in edata_bip")

    if bip_names is None:
        bip_names = [f"bip_{i}" for i in range(n_bip)]
    if len(bip_names) != n_bip:
        raise ValueError("bip_names length must match number of bipolar channels")

    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n_bip)))
    nrows = int(math.ceil(n_bip / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i in range(n_bip):
        ax = axes_flat[i]
        y = mean_signals[i, :]
        ax.plot(times_ms, y, color='k', lw=1)
        ax.set_title(bip_names[i], fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.set_xlim(times_ms.min(), times_ms.max())

    # hide unused axes
    for j in range(n_bip, len(axes_flat)):
        axes_flat[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return fig, mean_signals


# ----------------- Example usage (adapt to your chosen pairs) -----------------
# Define the bipolar pairs you want (anode, cathode, new_name)
bipolar_pairs = [
    ('EEG F3', 'EEG Fz', 'F3-Fz'),
    ('EEG Fz', 'EEG F4', 'Fz-F4'),
    ('EEG F4', 'EEG F8', 'F4-F8'),
    ('EEG T3', 'EEG C3', 'T3-C3'),
    ('EEG C3', 'EEG Cz', 'C3-Cz'),
    ('EEG Cz', 'EEG C4', 'Cz-C4'),
    ('EEG T4', 'EEG T5', 'T4-T5'),
    ('EEG P3', 'EEG Pz', 'P3-Pz'),
    ('EEG Fp1', 'EEG F3', 'Fp1-F3'),
    ('EEG F3', 'EEG T3', 'F3-T3'),
    ('EEG T3', 'EEG T4', 'T3-T4'),
    ('EEG Fpz', 'EEG T6', 'Fpz-T6'),
    ('EEG T6', 'EEG Fz', 'T6-Fz'),
    ('EEG Fz', 'EEG C3', 'Fz-C3'),
    ('EEG C3', 'EEG T5', 'C3-T5'),
    ('EEG Fp2', 'EEG F4', 'Fp2-F4'),
    ('EEG F7', 'EEG Cz', 'F7-Cz'),
    ('EEG Cz', 'EEG P3', 'Cz-P3'),
    ('EEG F7', 'EEG F8', 'F7-F8'),
    ('EEG F8', 'EEG C4', 'F8-C4'),
    ('EEG C4', 'EEG Cz', 'C4-Cz')
]

# bipolar_pairs = [('EEG F3', 'EEG Fz', 'F3-Fz'), ...] as you defined
edata_bip, bip_names, epochs_bip = create_bipolar_from_epochs(epochs, bipolar_pairs, return_epochsarray=True)
# times (ms)
times_ms = epochs.times * 1000.0
fig, means = plot_bipolar_averages(edata_bip*1000, times_ms, bip_names, ncols=3, figsize=(12,4), title="Bipolar averages")

# ====================== Apply montage =======================================

std = mne.channels.make_standard_montage('standard_1020')
std_pos = std.get_positions()['ch_pos']  # map like 'F3' -> (x,y,z)

ch_pos = {}
missing = []
for ch in epochs.ch_names:
    std_name = ch.replace('EEG ', '').strip()   # "EEG F3" -> "F3"
    if std_name in std_pos:
        ch_pos[ch] = std_pos[std_name]
    else:
        missing.append(ch)

print("Matched channels:", list(ch_pos.keys()))
if missing:
    print("WARNING - channels not found in standard_1020:", missing)

if len(ch_pos) == 0:
    raise RuntimeError("No channels matched the standard 10-20 names. Check channel labels.")

montage_subset = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

# 2) Attach montage to epochs (and to evoked if already created)
epochs.set_montage(montage_subset)

mne.viz.plot_sensors(epochs.info, show_names=True)
plt.title('Sensor positions (mapped to standard 10-20 where available)')
plt.show()

# PSD
epochs.average().compute_psd(
    method='welch', 
    fmin=0, 
    fmax=200).plot()

# TOPOMAPS
times_top = np.arange(0.0, 0.05, 0.005)
epochs.average().plot_topomap(times_top)

mne.viz.plot_epochs_image(
    epochs,
    sigma=0.5,
    vmin=-250,
    vmax=250,
    show=True,
)


# Quick checks / visualization
#  - show average evoked across epochs for first event type (if multiple)
epochs.average().plot(spatial_colors=True)   # interactive MNE plot

# Plotting mean signal for all channels
evoked = epochs.average()

# use evoked.data (n_ch x n_times) in Volts; convert to µV for plotting
data_uV = evoked.data * 1e6
times = evoked.times
ch_names = evoked.ch_names
n_ch = len(ch_names)

# colours for channels
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, n_ch))

# Create a figure and reserve room on the right for the inset
fig = plt.figure(figsize=(11, 5))

# Main axes: left, bottom, width, height (figure coords)
ax = fig.add_axes([0.06, 0.12, 0.72, 0.82])   # make width smaller than full figure so inset fits
lines = []
for i in range(n_ch):
    ln, = ax.plot(times, data_uV[i], color=colors[i], lw=1.2)
    lines.append(ln)

ax.set_xlim(times[0], times[-1])
ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)
ax.axhline(0.0, color='k', linestyle='-', alpha=0.4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('µV')
ax.set_title(f'EEG ({n_ch} channels) — mean across {len(epochs):d} epochs')

# Inset axes for head (positioned to the right, outside main trace area)
inset_ax = fig.add_axes([0.80, 0.62, 0.18, 0.30])  # tweak numbers to refine position & size
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_aspect('equal')

# Overlay channel NAMES at 2D layout positions, colored to match the traces
layout = mne.find_layout(evoked.info)
pos = np.asarray(layout.pos)   # may be (N,2) or (N,3)
layout_names = list(layout.names)
name_to_idx = {name: idx for idx, name in enumerate(layout_names)}

for i, ch in enumerate(ch_names):
    # find matching index in the layout (try exact, then stripped version)
    idx = name_to_idx.get(ch, name_to_idx.get(ch.replace('EEG ', '').strip(), None))
    if idx is None:
        # channel missing from layout (optional: print or skip)
        # print(f"Warning: channel {ch} not found in layout; skipping label.")
        continue

    coords = np.asarray(pos[idx])
    x, y = float(coords[0]), float(coords[1])  # robust to 2D/3D coords

    # plot a small colored dot (optional) so text stands out against head outline
    inset_ax.scatter(x, y, color=colors[i], edgecolor='k', s=30, zorder=11)

    # channel label (remove "EEG " prefix for compactness)
    label = ch.replace('EEG ', '').strip()

    # draw the text label in the same color as the trace
    # optional bbox can be used for readability; remove bbox=dict(...) if you prefer plain text
    inset_ax.text(x, y + 0.025, label, color=colors[i], fontsize=8,
                  ha='center', va='bottom', zorder=12,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

inset_ax.set_title('Channels', fontsize=9)
inset_ax.autoscale(False)  # keep inset fixed

# optionally add a legend of channel names + colors (placed between main axes and inset)
# fig.legend(lines, [ch.replace('EEG ', '') for ch in ch_names], loc='upper left',
#            bbox_to_anchor=(0.74, 0.96), fontsize='small', frameon=False, ncol=1)

plt.show()

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
    ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.set_title(ch_names[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times[0], times[-1])

for ax in axes[n_ch:]:
    ax.axis('off')

fig.suptitle("Per-channel mean across all epochs (µV)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()




def plot_all_epochs_per_channel(
    edata_sources=None,
    times=None,
    ch_names=None,
    sfreq=None,
    n_cols=6,
    epoch_alpha=0.25,
    epoch_lw=0.6,
    mean_color='C1',
    mean_lw=1.5,
    vmax_abs_convert=1e-2,
    figsize_per_col=(3, 2.2)
):
    """
    Plot all epochs per channel: each subplot contains all epochs for that channel (overlayed),
    plus the channel mean on top.

    Parameters
    ----------
    edata_sources : dict or None
        Optional dict of possible variable names to check for epoch data, e.g.
        {'edata': globals().get('edata'), 'edata_good': globals().get('edata_good'), ...}
        If None, the function will try common global names itself.
        Accepted epoch array shape: (n_epochs, n_channels, n_times)
    times : ndarray or None
        Time vector (n_times,) in seconds. If None, will attempt to use epochs.times or construct from sfreq.
    ch_names : list or None
        Channel names (length n_channels). If None, will try epochs.ch_names or create generic names.
    sfreq : float or None
        Sampling frequency (Hz) — used only if times is None to synthesize a time vector.
    n_cols : int
        Number of subplot columns.
    epoch_alpha, epoch_lw : float
        Alpha and linewidth for individual epoch traces.
    mean_color, mean_lw : color/float
        Color and linewidth for the averaged trace plotted on top.
    vmax_abs_convert : float
        Threshold to decide if data is in Volts (abs max < threshold) — then convert to µV.
    figsize_per_col : tuple
        Width/height multipliers per column used to build figsize.
    """
    # Try to locate epoch data if not provided explicitly
    edata = None
    if edata_sources is None:
        # common names used in this session
        candidates = ['edata', 'edata_uV', 'edata_good', 'edata_baselined', 'edata_cropped_all', 'epochs_data']
        g = globals()
        for name in candidates:
            if name in g and g[name] is not None:
                arr = g[name]
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    edata = arr
                    break
        # fallback to epochs object if present
        if edata is None and 'epochs' in g:
            try:
                edata = g['epochs'].get_data()
            except Exception:
                edata = None
    else:
        # check provided dict for valid arrays
        for name, arr in edata_sources.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                edata = arr
                break

    if edata is None:
        raise RuntimeError("No epoch data found. Provide edata (n_epochs,n_ch,n_times) or an epochs object.")

    n_epochs, n_ch, n_times = edata.shape

    # Determine times vector
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
                # attempt to get sfreq from epochs.info if available
                if 'epochs' in g and hasattr(g['epochs'], 'info'):
                    sfreq = g['epochs'].info.get('sfreq', None)
            if sfreq is None:
                raise RuntimeError("No time vector available and sfreq not provided.")
            times = np.arange(n_times) / float(sfreq)

    # Determine channel names
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

    # Convert to µV if values are in Volts (heuristic)
    if np.nanmax(np.abs(edata)) < vmax_abs_convert:
        edata_uV = edata * 1e6
    else:
        edata_uV = edata.copy()

    # Prepare plotting grid
    n_rows = int(math.ceil(n_ch / n_cols))
    figsize = (figsize_per_col[0] * n_cols, figsize_per_col[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.ravel()

    for ch in range(n_ch):
        ax = axes[ch]
        # Plot all epochs for this channel
        for ep in range(n_epochs):
            ax.plot(times, edata_uV[ep, ch, :], color='C0', alpha=epoch_alpha, lw=epoch_lw)

        # Plot mean on top
        mean_trace = np.nanmean(edata_uV[:, ch, :], axis=0)
        ax.plot(times, mean_trace, color=mean_color, lw=mean_lw, label='Mean')

        ax.axvline(0.0, color='k', linestyle='--', alpha=0.6)   # event/stimulus onset
        ax.set_title(ch_names[ch], fontsize=8)
        if ch % n_cols == 0:
            ax.set_ylabel('µV')
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, alpha=0.25)
        if ch == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Turn off unused axes
    for ax in axes[n_ch:]:
        ax.axis('off')

    fig.suptitle("All epochs per channel (overlay) with mean in color", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


plot_all_epochs_per_channel(edata_sources={'edata': edata}, times=times, ch_names=ch_names, sfreq=sfreq)



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
    edata_sources={'edata': edata},
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
edata_cropped_all = edata_bip[:, :, crop_samples:] * 1e6
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
#eeg_labels = list(epochs.ch_names)
eeg_labels = bip_names
# Plot evoked potential
n_ch = evoked_potential.shape[0]
n_cols = 6
n_rows = int(np.ceil(n_ch / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.2*n_rows), sharex=True)
axes = axes.ravel()

for i in range(n_ch):
    ax = axes[i]
    ax.plot(times_cropped_all, evoked_potential[i], color='C0', lw=1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset')
    ax.set_title(eeg_labels[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times_cropped_all[0], times_cropped_all[-1])
    ax.set_xlabel('Time (ms)')
    ax.grid(True, alpha=0.3)

for ax in axes[n_ch:]:
    ax.axis('off')

fig.suptitle(f"Evoked Potential After Stimulus Cropping\n({edata.shape[0]} good epochs, stimulus artifact removed)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ================== PLOT COMPARISON ==================
print("\nPlotting comparison: before vs after cropping...")

# Show first channel as example
ch_idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# Before cropping
mean_before = np.mean(edata_bip[:, ch_idx, :]*1e6, axis=0)
ax1.plot(times, mean_before, color='C0', lw=1.5)
ax1.axvspan(0, crop_duration_s, alpha=0.3, color='red', label=f'Removed: {crop_duration_ms} ms')
ax1.axvline(0, color='k', linestyle='--', alpha=0.6)
ax1.set_ylabel('µV')
ax1.set_title(f'{eeg_labels[ch_idx]} - Before Cropping')
ax1.legend()
ax1.grid(True, alpha=0.3)

# After cropping
mean_after = np.mean(edata_cropped_all[:, ch_idx, :], axis=0)
ax2.plot(times_cropped_all, mean_after, color='C1', lw=1.5)
ax2.axvline(0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset + artifact')
ax2.set_xlabel('Time (s)')
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
    ch_names=bip_names,
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
    channel_names=bip_names,
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

# ================== GROUP AVERAGING ==================
print("\n" + "="*50)
print("AVERAGING EPOCHS IN GROUPS OF 40")
print("="*50)

group_size = 30
n_good_epochs = edata_cropped_all.shape[0]
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
    group_data = edata_cropped_all[start_idx:end_idx]
    group_mean = np.mean(group_data, axis=0)  # shape: (n_channels, n_times)
    
    grouped_averages.append(group_mean)
    group_info.append({
        'group_num': group_idx + 1,
        'start_epoch': start_idx,
        'end_epoch': end_idx,
        'n_epochs': group_size
    })
    
    
    
    print(f"  Group {group_idx + 1}: epochs {start_idx}-{end_idx-1}")

# Handle remaining epochs if any
if remainder > 0:
    start_idx = n_groups * group_size
    end_idx = n_good_epochs
    
    group_data = edata_cropped_all[start_idx:end_idx]
    group_mean = np.mean(group_data, axis=0)
    
    grouped_averages.append(group_mean)
    group_info.append({
        'group_num': n_groups + 1,
        'start_epoch': start_idx,
        'end_epoch': end_idx,
        'n_epochs': remainder})
    
    print(f"  Group {n_groups + 1} (incomplete): epochs {start_idx}-{end_idx-1} ")

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
    title = (f"Group {info['group_num']}: Epochs {info['start_epoch']}-{info['end_epoch']-1} "
             f"(n={info['n_epochs']})")
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

print(f"\n✓ Generated {len(grouped_averages)} plots")


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
for ax in axes[n_channels:]:
    ax.axis('off')

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(grouped_averages))

fig.suptitle("All Groups Overlay - Per Channel Average", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()



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
        'n_epochs': info['n_epochs']
    }
    for info in group_info
}

with open(f"{output_dir}/group_info.json", 'w') as f:
    json.dump(group_info_json, f, indent=2)

print(f"✓ Saved to {output_dir}/")
print(f"  - grouped_averages.npy: shape {grouped_averages.shape}")
print(f"  - peak_amplitudes.npy: shape {peak_amplitudes.shape}")
print(f"  - group_info.json: group metadata")


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


# --------------------- Detector (copied & used as-is) ---------------------
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

