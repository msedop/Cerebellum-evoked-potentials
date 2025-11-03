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
file_path = r"C:\Users\msedo\Documents\Cerebelo\Chimula Mark\13.14_stim_2.edf"  # Change this to your file path

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
raw.drop_channels( ['Chin+','ECG+','DI+','DD+','RONQ+','CAN+','TER+','PCO2+','EtCO+','Pos+','Tor+','Abd+','TIBI+','TIBD+','thor+','abdo+','PULS+','BEAT+','SpO2+','MKR+'])
raw.drop_channels([
   'EEG Fp1', 'EEG Fp2', 'EEG EOGI', 'EEG T3', 'EEG T4',
   'EEG O1', 'EEG EOGD', 'EEG O2', 'EEG A1', 'EEG A2', 'EEG EOGX'
])

raw.drop_channels([
    'EEG F4', 'EEG Cz'
])


mapping = {
    'EEG F3': 'EEG F1',
    'EEG F7': 'EEG F3',
    'EEG C3' : 'EEG C1',
    'EEG P3' : 'EEG CP1',
    'EEG T5' : 'EEG CP3',
    'EEG F4' : 'EEG F2',
    'EEG F8' : 'EEG F4',
    'EEG C4' : 'EEG C2',
    'EEG P4' : 'EEG CP2',
    'EEG T6' : 'EEG CP4'

}

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
epoch_tmax = 0.2

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

# # ========================== Stimulus artifact subtraction ====================

# # Helper: define artifact window and taper
# ms_window = 50.0                          # artifact duration in milliseconds (change if needed)
# artifact_mask = (times * 1000.0 >= 0) & (times * 1000.0 <= ms_window)   # boolean mask over time samples

# # optional taper to avoid steps (Hann / cosine taper over artifact edges)
# def make_taper(mask, taper_len_samples=5):
#     """
#     mask : boolean array over time
#     taper_len_samples : how many samples to taper at each edge (int)
#     returns a 1D array (n_times,) with 1 inside mask and taper to 0 at edges
#     """
#     n = len(mask)
#     w = np.zeros(n, dtype=float)
#     idx = np.where(mask)[0]
#     if idx.size == 0:
#         return w
#     start, end = idx[0], idx[-1]
#     core = np.arange(start + taper_len_samples, end - taper_len_samples + 1)
#     if core.size > 0:
#         w[core] = 1.0
#     # left taper
#     left = np.arange(start, min(start + taper_len_samples, end + 1))
#     if left.size:
#         w[left] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, left.size)))
#     # right taper
#     right = np.arange(max(end - taper_len_samples + 1, start), end + 1)
#     if right.size:
#         w[right] = 0.5 * (1 - np.cos(np.linspace(np.pi, 2*np.pi, right.size)))
#     return w

# taper = make_taper(artifact_mask, taper_len_samples=max(1, int(0.002 * sfreq)))  # e.g., 2 ms taper

# # edata: shape (n_epochs, n_channels, n_times) in Volts
# edata = epochs.get_data()   # ensure you use the current epochs
# n_epochs, n_ch, n_t = edata.shape

# # compute per-channel artifact as mean across epochs (shape: n_channels x n_times)
# artifact_per_channel = np.nanmean(edata[:, :, :], axis=0)  # (n_channels, n_times)

# # if you want to only remove in artifact window, multiply artifact by mask/taper
# artifact_mask_2d = artifact_per_channel * taper[np.newaxis, :]   # shape (n_channels, n_times)

# # subtract artifact from each epoch (channel-wise)
# edata_corrected = edata - artifact_mask_2d[np.newaxis, :, :]

# # build a new EpochsArray with same info, events, event_id, tmin
# epochs_clean = mne.EpochsArray(edata_corrected, info=epochs.info.copy(), tmin=epochs.tmin,
#                                events=getattr(epochs, 'events', None),
#                                event_id=getattr(epochs, 'event_id', None), verbose=False)

# # quick check: compare evoked before/after
# evoked_before = epochs.average()
# evoked_after = epochs_clean.average()

# # plot comparison for a channel index (e.g., 0)
# ch_idx = 0
# plt.figure(figsize=(8,4))
# plt.plot(times*1000, evoked_before.data[ch_idx]*1e6, label='Before (µV)')
# plt.plot(times*1000, evoked_after.data[ch_idx]*1e6, label='After (µV)')
# plt.axvspan(0, ms_window, color='red', alpha=0.1, label='Artifact window')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude (µV)')
# plt.title(f'Channel {epochs.ch_names[ch_idx]} - evoked before vs after artifact subtraction')
# plt.legend()
# plt.show()


# # use cleaned epochs to compute downstream results
# edata_clean = epochs_clean_scaled.get_data()   # or epochs_clean.get_data() etc.

# # If you later crop the first N samples, do it after cleaning
# crop_duration_ms = 60
# crop_samples = int(round(crop_duration_ms / 1000.0 * sfreq))
# edata_cropped_all = edata_clean[:, :, crop_samples:] * 1e6   # use µV for your other code
# times_cropped_all = epochs_clean_scaled.times[crop_samples:] - epochs_clean_scaled.times[crop_samples]  # start at 0.0

# ================== CROP STIMULUS ARTIFACT ==================
print("\n" + "="*50)
print("CROPPING STIMULUS ARTIFACT")
print("="*50)

# Define stimulus artifact duration
crop_duration_ms = 100 # should be 10.5
crop_duration_s = crop_duration_ms / 1000.0
crop_samples = int(np.round(crop_duration_s * sfreq))

print(f"Removing stimulus artifact: {crop_duration_ms} ms ({crop_samples} samples)")

# Remove the first N samples (stimulus artifact period)
# After removal: data starts at ~10.5 ms after stimulus onset
edata_cropped_all = edata[:, :, crop_samples:] * 1e6
times_cropped_all = times[crop_samples:]

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
    ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.axvline(0.0, color='k', linestyle='--', alpha=0.6, label='Stimulus onset')
    ax.set_title(eeg_labels[i], fontsize=8)
    if i % n_cols == 0:
        ax.set_ylabel('µV')
    ax.set_xlim(times_cropped_all[0], times_cropped_all[-1])
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
mean_before = np.mean(edata[:, ch_idx, :]*1e6, axis=0)
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
    ch_names=ch_names,
    mean_color='tab:blue',
    envelope_kind='std',
    envelope_alpha=0.35,
    envelope_lighten_frac=0.50,
    return_fig_ax=False
)

# =================== Peak Detection ====================================


try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None  # fallback: no smoothing if scipy not available


def _ensure_times_ms(times: np.ndarray, n_times: int) -> np.ndarray:
    times = np.asarray(times)
    if times.size != n_times:
        # if lengths mismatch, resample times linearly to match n_times
        times = np.linspace(float(times[0]), float(times[-1]), n_times)
    # convert seconds -> ms if values look like seconds (max <= 1000)
    if np.nanmax(np.abs(times)) <= 1000:
        return times * 1000.0
    return times.copy()


def _smooth_trace(trace: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples is None or sigma_samples <= 0:
        return trace
    if gaussian_filter1d is None:
        # scipy not available: do a very simple moving average as fallback
        w = max(3, int(round(sigma_samples * 2 + 1)))
        kernel = np.ones(w) / float(w)
        return np.convolve(trace, kernel, mode='same')
    else:
        return gaussian_filter1d(trace, sigma=sigma_samples, mode='mirror')


def find_first_n_turning_points_per_channel(
    edata: Optional[np.ndarray] = None,
    epochs=None,
    evoked=None,
    times: Optional[np.ndarray] = None,
    ch_names: Optional[List[str]] = None,
    n_peaks: int = 4,
    smooth_sigma_ms: Optional[float] = 2.0,
    min_peak_distance_ms: float = 8.0,
    prominence_abs: Optional[float] = None,
    time_window_ms: Optional[Tuple[float, float]] = None,
    invert_set: Optional[List[str]] = None,
    plot_results: bool = False,
    return_dataframe: bool = True
) -> Union[pd.DataFrame, dict]:
    """
    Detect turning points (first derivative zero crossings) and return the first
    n_peaks (time-ordered) for each channel.

    Parameters
    ----------
    edata : ndarray (n_epochs, n_ch, n_times) or None
        Epoch data. If None, provide `epochs` or `evoked`.
    epochs : mne.Epochs-like (optional)
        Used to get data and times if edata is None.
    evoked : mne.Evoked-like (optional)
        Used if provided (averaged data).
    times : array-like (n_times,) in seconds or ms. Required if edata is provided.
    ch_names : list of channel names (n_ch,)
    n_peaks : int
        Number of earliest turning points (per channel) to return.
    smooth_sigma_ms : float or None
        Gaussian smoothing sigma (ms) applied to the trace before derivative.
    min_peak_distance_ms : float
        Minimum spacing (ms) enforced between reported turning points (per channel).
    prominence_abs : float or None
        Minimum absolute amplitude difference (µV) to accept a turning point.
        If None a heuristic of 5% of peak-to-peak (or 1 µV) is used per channel.
    time_window_ms : (tmin_ms, tmax_ms) or None
        If provided, restrict search to this window.
    plot_results : bool
        If True, plot the mean traces and annotate found turning points.
    return_dataframe : bool
        If True return a pandas.DataFrame, else return a dict mapping channel -> list.

    Returns
    -------
    pandas.DataFrame or dict
    """

    edata = np.asarray(edata)

    n_epochs, n_ch, n_times = edata.shape

    times_ms = _ensure_times_ms(np.asarray(times), n_times)
    dt_ms = np.mean(np.diff(times_ms))

    # channel names
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]
    if len(ch_names) != n_ch:
        raise ValueError("ch_names length mismatch with data channels")

    # unit conversion: assume Volts if small values -> convert to µV
    if np.nanmax(np.abs(edata)) < 1e-2:
        edata_uV = edata * 1e6
    else:
        edata_uV = edata.copy()

    # compute mean trace per channel
    mean_traces = np.nanmean(edata_uV, axis=0)  # (n_ch, n_times)

    # restrict to time window
    if time_window_ms is not None:
        tmin_ms, tmax_ms = time_window_ms
        win_mask = (times_ms >= tmin_ms) & (times_ms <= tmax_ms)
        if not np.any(win_mask):
            raise ValueError("time_window_ms does not overlap times.")
        win_idx = np.where(win_mask)[0]
        win_start, win_end = win_idx[0], win_idx[-1] + 1
    else:
        win_start, win_end = 0, n_times

    results = []
    results_dict = {}

    min_dist_pts = max(1, int(round(min_peak_distance_ms / dt_ms)))
    # smoothing sigma in samples
    sigma_samples = None
    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        sigma_samples = max(0.5, float(smooth_sigma_ms / dt_ms))

    for ch in range(n_ch):
        trace = mean_traces[ch, :]
        
        # Decide whether to invert this channel for detection
        invert = (ch_names[ch] in invert_set)
        
        # Build detection trace. If invert==True we multiply by -1 so troughs become peaks.
        trace = -trace if invert else trace
        
        # smooth trace (helps derivative stability)
        trace_s = _smooth_trace(trace, sigma_samples) if sigma_samples is not None else trace

        # restrict window
        t_win = times_ms[win_start:win_end]
        tr_win = trace_s[win_start:win_end]

        # derivative (dX/dt) in µV per ms
        deriv = np.gradient(tr_win, dt_ms)

        # find zero-crossings of derivative: where sign changes
        # Avoid sign(0) issues by adding small eps to derivative
        eps = 1e-12
        s = np.sign(deriv + eps)
        sign_changes = np.where(s[:-1] * s[1:] < 0)[0]  # indices i where sign changes between i and i+1
        # sign change at index i corresponds to a turning between samples i and i+1.
        # We'll refine by searching local extremum in a small neighborhood around i+1
        candidates = []
        for i in sign_changes:
            # convert to absolute index in original trace
            center_idx = win_start + (i + 1)
            # neighborhood ± min_dist_pts (but limited)
            left = max(win_start, center_idx - min_dist_pts)
            right = min(win_end - 1, center_idx + min_dist_pts)
            # find true extremum in original (unsmoothed) trace for amplitude accuracy
            seg = trace[left:right + 1]
            if deriv[i] > 0 and deriv[i + 1] < 0:
                # positive to negative => local maximum
                rel = int(np.argmax(seg))
                peak_idx = left + rel
                polarity = 'pos'
                amp = float(-trace[peak_idx]) if invert else float(trace[peak_idx])
            else:
                # unexpected (shouldn't happen), skip
                continue
            candidates.append((int(peak_idx), amp, polarity))

        # deduplicate very close candidates (keep the earliest)
        if candidates:
            # sort by index (time)
            candidates = sorted(candidates, key=lambda x: x[0])
            deduped = []
            last_idx = -9999
            for idx_abs, amp, pol in candidates:
                if idx_abs - last_idx <= min_dist_pts:
                    # too close to previous; keep the one with larger abs amplitude
                    prev_idx, prev_amp, prev_pol = deduped[-1]
                    if abs(amp) > abs(prev_amp):
                        deduped[-1] = (idx_abs, amp, pol)
                        last_idx = idx_abs
                    else:
                        # keep previous, skip current
                        continue
                else:
                    deduped.append((idx_abs, amp, pol))
                    last_idx = idx_abs
            candidates = deduped

        # apply prominence/absolute amplitude threshold
        if prominence_abs is None:
            # heuristic: 5% of peak-to-peak in the search window but at least 1 µV
            p2p = np.nanmax(tr_win) - np.nanmin(tr_win)
            prom_thresh = max(1.0, 0.05 * p2p)
        else:
            prom_thresh = float(prominence_abs)

        filtered = []
        for idx_abs, amp, pol in candidates:
            if abs(amp) >= prom_thresh:
                filtered.append((idx_abs, amp, pol))
        # keep first n_peaks in time order; pad if fewer
        peaks_for_channel = []
        for rank in range(n_peaks):
            if rank < len(filtered):
                idx_abs, amp, pol = filtered[rank]
                latency = float(times_ms[idx_abs])
                peaks_for_channel.append((rank + 1, latency, amp, pol, int(idx_abs)))
                results.append({
                    'channel': ch_names[ch],
                    'peak_rank': rank + 1,
                    'latency_ms': latency,
                    'amplitude_uV': amp,
                    'polarity': pol,
                    'index': int(idx_abs)
                })
            else:
                peaks_for_channel.append((rank + 1, np.nan, np.nan, None, None))
                results.append({
                    'channel': ch_names[ch],
                    'peak_rank': rank + 1,
                    'latency_ms': np.nan,
                    'amplitude_uV': np.nan,
                    'polarity': None,
                    'index': None
                })
        results_dict[ch_names[ch]] = peaks_for_channel
        invert = False

    df = pd.DataFrame(results, columns=['channel', 'peak_rank', 'latency_ms', 'amplitude_uV', 'polarity', 'index'])
    
    if plot_results:
        n_cols = min(6, n_ch)
        n_rows = int(np.ceil(n_ch / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows), squeeze=False)
        axs = axs.reshape(-1)
        for i, ch in enumerate(range(n_ch)):
            ax = axs[i]
            ax.plot(times_ms, mean_traces[ch, :], color='C0', lw=1.2)
            pe_list = results_dict[ch_names[ch]]
            for (_, lat, amp, pol, idx_abs) in pe_list:
                if idx_abs is None:
                    continue
                color = 'tab:green' if pol == 'pos' else 'tab:red'
                ax.plot(times_ms[idx_abs], amp, marker='o', color=color)
                ax.text(times_ms[idx_abs], amp, f"{lat:.0f} ms", fontsize=7, ha='left', va='bottom')
            ax.set_title(ch_names[ch], fontsize=9)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('µV')
            ax.grid(alpha=0.2)
        # hide unused subplots
        for j in range(n_ch, len(axs)):
            axs[j].axis('off')
        plt.tight_layout()

    if return_dataframe:
        return df
    else:
        return results_dict



df = find_first_n_turning_points_per_channel(
    edata=edata_cropped_all,
    times=times_cropped_all,
    ch_names=ch_names,
    n_peaks=4,
    smooth_sigma_ms=1.0,
    min_peak_distance_ms=5.0,
    prominence_abs=None,
    time_window_ms=(0, 200),
    invert_set = ['EEG F3', 'EEEG F1', 'EEG Fz', 'EEG F2', 'EEG F4'],
    plot_results=True,
    return_dataframe=True
)
print(df)




"""
find_first_n_turning_points_with_table_labels.py

Detect turning points (first-derivative zero crossings) and return the first
n_peaks (time-ordered) for each channel. For plotting, the function now:

- Adds small text labels on top of each detected peak following the natural order
  of appearance: N0, N1, N2, N3 (N(index-1) for peak_rank index).
- Adds a compact summary "table" at the top-right of each subplot listing for
  each labeled peak its latency and amplitude (units: ms, µV). Missing peaks
  are shown as "--".

The function otherwise preserves the prior behavior:
- smoothing before derivative,
- extracting amplitude from original (unsmoothed) mean trace,
- optional baseline-window amplitude subtraction (amplitude_from_baseline_uV),
- returns a pandas DataFrame and optionally plots the annotated figure.

Usage:
    df = find_first_n_turning_points_per_channel(
        edata=edata_cropped_all,
        times=times_cropped_all,
        ch_names=ch_names,
        n_peaks=4,
        smooth_sigma_ms=1.0,
        min_peak_distance_ms=5.0,
        prominence_abs=None,
        time_window_ms=(0,200),
        baseline_window_ms=(-200,0),
        plot_results=True,
        return_dataframe=True
    )
"""
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None  # fallback: no smoothing if scipy not available


def _ensure_times_ms(times: np.ndarray, n_times: int) -> np.ndarray:
    times = np.asarray(times)
    if times.size != n_times:
        # if lengths mismatch, resample times linearly to match n_times
        times = np.linspace(float(times[0]), float(times[-1]), n_times)
    # convert seconds -> ms if values look like seconds (max <= 1000)
    if np.nanmax(np.abs(times)) <= 1000:
        return times * 1000.0
    return times.copy()


def _smooth_trace(trace: np.ndarray, sigma_samples: Optional[float]) -> np.ndarray:
    if sigma_samples is None or sigma_samples <= 0:
        return trace
    if gaussian_filter1d is None:
        # scipy not available: do a very simple moving average as fallback
        w = max(3, int(round(sigma_samples * 2 + 1)))
        kernel = np.ones(w) / float(w)
        return np.convolve(trace, kernel, mode='same')
    else:
        return gaussian_filter1d(trace, sigma=sigma_samples, mode='mirror')


def find_first_n_turning_points_per_channel(
    edata: Optional[np.ndarray] = None,
    epochs=None,
    evoked=None,
    times: Optional[np.ndarray] = None,
    ch_names: Optional[List[str]] = None,
    n_peaks: int = 4,
    smooth_sigma_ms: Optional[float] = 2.0,
    min_peak_distance_ms: float = 8.0,
    prominence_abs: Optional[float] = None,
    time_window_ms: Optional[Tuple[float, float]] = None,
    baseline_window_ms: Optional[Tuple[float, float]] = None,
    invert_set: Optional[List[str]] = None,
    plot_results: bool = False,
    return_dataframe: bool = True
) -> Union[pd.DataFrame, Dict[str, List[Tuple[int, float, float, float, Optional[str], Optional[int]]]]]:
    """
    Detect turning points (first derivative zero crossings) and return the first
    n_peaks (time-ordered) for each channel. Also compute amplitudes for each peak
    and annotate plots with N0..N3 labels plus a small summary table.

    New parameter:
      - baseline_window_ms: optional (tmin_ms, tmax_ms). If provided, compute the mean of the
        channel mean trace within this window and report amplitude_from_baseline_uV = peak - baseline_mean.
        If None, amplitude_from_baseline_uV is set to np.nan.

    Returns DataFrame columns:
      ['channel', 'peak_rank', 'latency_ms', 'amplitude_uV', 'amplitude_from_baseline_uV', 'polarity', 'index']
    """

    edata = np.asarray(edata)
    if edata.ndim != 3:
        raise ValueError("edata must be 3D: (n_epochs, n_ch, n_times)")

    n_epochs, n_ch, n_times = edata.shape

    if times is None:
        raise RuntimeError("times vector is required (pass times or an epochs/evoked object).")

    times_ms = _ensure_times_ms(np.asarray(times), n_times)
    dt_ms = np.mean(np.diff(times_ms))

    # channel names
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]
    if len(ch_names) != n_ch:
        raise ValueError("ch_names length mismatch with data channels")

    # unit conversion: assume Volts if small values -> convert to µV
    if np.nanmax(np.abs(edata)) < 1e-2:
        edata_uV = edata * 1e6
    else:
        edata_uV = edata.copy()

    # compute mean trace per channel
    mean_traces = np.nanmean(edata_uV, axis=0)  # (n_ch, n_times)

    # compute baseline means if requested (from original mean traces, not smoothed)
    if baseline_window_ms is not None:
        bmin, bmax = baseline_window_ms
        bmask = (times_ms >= bmin) & (times_ms <= bmax)
        if not np.any(bmask):
            raise ValueError("baseline_window_ms does not overlap times.")
        baseline_means = np.nanmean(mean_traces[:, bmask], axis=1)  # shape (n_ch,)
    else:
        baseline_means = np.full((n_ch,), np.nan)

    # restrict to time window
    if time_window_ms is not None:
        tmin_ms, tmax_ms = time_window_ms
        win_mask = (times_ms >= tmin_ms) & (times_ms <= tmax_ms)
        if not np.any(win_mask):
            raise ValueError("time_window_ms does not overlap times.")
        win_idx = np.where(win_mask)[0]
        win_start, win_end = win_idx[0], win_idx[-1] + 1
    else:
        win_start, win_end = 0, n_times

    results = []
    results_dict = {}

    min_dist_pts = max(1, int(round(min_peak_distance_ms / dt_ms)))
    # smoothing sigma in samples
    sigma_samples = None
    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        sigma_samples = max(0.5, float(smooth_sigma_ms / dt_ms))

    for ch in range(n_ch):
        trace = mean_traces[ch, :]
        
        # Decide whether to invert this channel for detection
        invert = (ch_names[ch] in invert_set)
        
        # Build detection trace. If invert==True we multiply by -1 so troughs become peaks.
        trace = -trace if invert else trace
        
        
        # smooth trace (helps derivative stability) - derivative computed on smoothed trace
        trace_s = _smooth_trace(trace, sigma_samples) if sigma_samples is not None else trace

        # restrict window (operate on smoothed trace for derivative, but amplitudes read from original trace)
        t_win = times_ms[win_start:win_end]
        tr_win = trace_s[win_start:win_end]
        tr_orig_win = trace[win_start:win_end]  # original unsmoothed window for amplitude reading

        # derivative (dX/dt) in µV per ms
        deriv = np.gradient(tr_win, dt_ms)

        # find zero-crossings of derivative: sign-change detection (robustness to zero)
        small_eps = 1e-12
        s = np.sign(deriv + small_eps)
        sign_changes = np.where(s[:-1] * s[1:] < 0)[0]  # indices i where sign changes between i and i+1

        candidates = []
        for i in sign_changes:
            # convert to absolute index in original trace
            center_idx = win_start + (i + 1)
            # neighborhood ± min_dist_pts (but limited)
            left = max(win_start, center_idx - min_dist_pts)
            right = min(win_end - 1, center_idx + min_dist_pts)
            # find true extremum in original (unsmoothed) trace for amplitude accuracy
            seg = trace[left:right + 1]
            if deriv[i] > 0 and deriv[i + 1] < 0:
                # positive to negative => local maximum
                rel = int(np.argmax(seg))
                peak_idx = left + rel
                polarity = 'pos'
            else:
                # unexpected (shouldn't happen), skip
                continue
            # amplitude from original unsmoothed mean trace
            amp = float(-trace[peak_idx]) if invert else float(trace[peak_idx])
            candidates.append((int(peak_idx), amp, polarity))

        # deduplicate very close candidates (keep the earliest/largest)
        if candidates:
            candidates = sorted(candidates, key=lambda x: x[0])
            deduped = []
            last_idx = -9999
            for idx_abs, amp, pol in candidates:
                if idx_abs - last_idx <= min_dist_pts:
                    prev_idx, prev_amp, prev_pol = deduped[-1]
                    if abs(amp) > abs(prev_amp):
                        deduped[-1] = (idx_abs, amp, pol)
                        last_idx = idx_abs
                    else:
                        continue
                else:
                    deduped.append((idx_abs, amp, pol))
                    last_idx = idx_abs
            candidates = deduped

        # apply prominence/absolute amplitude threshold (using smoothed window amplitude range if heuristic)
        if prominence_abs is None:
            # heuristic: 5% of peak-to-peak in the smoothed search window but at least 1 µV
            p2p = np.nanmax(tr_win) - np.nanmin(tr_win)
            prom_thresh = max(1.0, 0.05 * p2p)
        else:
            prom_thresh = float(prominence_abs)

        filtered = []
        for idx_abs, amp, pol in candidates:
            if abs(amp) >= prom_thresh:
                filtered.append((idx_abs, amp, pol))

        # keep first n_peaks in time order; compute amplitude_from_baseline too
        peaks_for_channel = []
        for rank in range(n_peaks):
            if rank < len(filtered):
                idx_abs, amp, pol = filtered[rank]
                latency = float(times_ms[idx_abs])
                # amplitude from original mean trace (already amp)
                amplitude_uV = amp
                # amplitude relative to baseline (if baseline provided)
                baseline_mean = baseline_means[ch]
                if not np.isnan(baseline_mean):
                    amplitude_from_baseline_uV = amplitude_uV - float(baseline_mean)
                else:
                    amplitude_from_baseline_uV = np.nan
                peaks_for_channel.append((rank + 1, latency, amplitude_uV, amplitude_from_baseline_uV, pol, int(idx_abs)))
                results.append({
                    'channel': ch_names[ch],
                    'peak_rank': rank + 1,
                    'latency_ms': latency,
                    'amplitude_uV': amplitude_uV,
                    'amplitude_from_baseline_uV': amplitude_from_baseline_uV,
                    'polarity': pol,
                    'index': int(idx_abs)
                })
            else:
                peaks_for_channel.append((rank + 1, np.nan, np.nan, np.nan, None, None))
                results.append({
                    'channel': ch_names[ch],
                    'peak_rank': rank + 1,
                    'latency_ms': np.nan,
                    'amplitude_uV': np.nan,
                    'amplitude_from_baseline_uV': np.nan,
                    'polarity': None,
                    'index': None
                })
        results_dict[ch_names[ch]] = peaks_for_channel
        invert = False
        
    df = pd.DataFrame(results, columns=['channel', 'peak_rank', 'latency_ms', 'amplitude_uV', 'amplitude_from_baseline_uV', 'polarity', 'index'])

    if plot_results:
        n_cols = min(6, n_ch)
        n_rows = int(np.ceil(n_ch / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows), squeeze=False)
        axs = axs.reshape(-1)

        for i, ch in enumerate(range(n_ch)):
            ax = axs[i]
            ax.plot(times_ms, mean_traces[ch, :], color='C0', lw=1.2)
            pe_list = results_dict[ch_names[ch]]
            # determine a small vertical offset for labels (5% of y-range)
            y_min, y_max = np.nanmin(mean_traces[ch, :]), np.nanmax(mean_traces[ch, :])
            y_range = y_max - y_min if (y_max > y_min) else 1.0
            y_offset = 0.05 * y_range

            # build summary lines for the "table" at top-right
            lines = []
            for (rank, latency, amp, amp_from_base, pol, idx_abs) in pe_list:
                label_name = f"N{rank-1}"
                if idx_abs is None:
                    lines.append(f"{label_name}: --")
                else:
                    if not np.isnan(amp_from_base):
                        lines.append(f"{label_name}: {int(round(latency))} ms, {amp_from_base:.1f} µV")
                    else:
                        lines.append(f"{label_name}: {int(round(latency))} ms, {amp:.1f} µV")

            # plot peak markers and small peak labels N0..N3 above the peaks
            for (rank, latency, amp, amp_from_base, pol, idx_abs) in pe_list:
                if idx_abs is None:
                    continue
                label_name = f"N{rank-1}"
                color = 'tab:green' if pol == 'pos' else 'tab:red'
                ax.plot(times_ms[idx_abs], amp, marker='o', color=color)
                # place the small name above the marker
                ax.text(times_ms[idx_abs], amp + y_offset, label_name,
                        fontsize=8, ha='center', va='bottom', fontweight='bold', color=color)

            # draw the small summary table in top-right using axes coordinates
            table_text = "\n".join(lines)
            ax.text(0.98, 0.98, table_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

            ax.set_title(ch_names[ch], fontsize=9)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('µV')
            ax.grid(alpha=0.2)

        # hide unused subplots
        for j in range(n_ch, len(axs)):
            axs[j].axis('off')
        plt.tight_layout()

    if return_dataframe:
        return df
    else:
        return results_dict
    
    
df = find_first_n_turning_points_per_channel(
    edata=edata_cropped_all,
    times=times_cropped_all,
    ch_names=ch_names,
    n_peaks=4,
    smooth_sigma_ms=1.5,
    min_peak_distance_ms=5.0,
    prominence_abs=None,
    time_window_ms=(0, 200),    
    invert_set = ['EEG F3', 'EEEG F1', 'EEG Fz', 'EEG F2', 'EEG F4'],
    plot_results=True,
    return_dataframe=True
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

group_size = 100
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
    
    find_first_n_turning_points_per_channel(
    edata=group_data,
    times=times_cropped_all,
    ch_names=ch_names,
    n_peaks=4,
    smooth_sigma_ms=1,
    min_peak_distance_ms=5.0,
    prominence_abs=None,
    time_window_ms=(0, 200),
    invert_set = ['EEG F3', 'EEEG F1', 'EEG Fz', 'EEG F2', 'EEG F4'],
    plot_results=True,
    return_dataframe=True
    )
    
    
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
    print(f"{info['group_num']:<8} {epoch_range:<15} {info['n_epochs']:<5}")
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


# ===================== ALIGNMENT OF EPOCHS ============================

# -------- PARAMETERS / fallback ignore list (same as before) ----------
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
