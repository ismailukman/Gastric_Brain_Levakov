import os
import sys
import pathlib

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from scipy.signal import resample, hilbert
from mne.filter import filter_data
import matplotlib.pyplot as plt

##############################################################################
# Configuration (edit these paths and parameters to match your environment)  #
##############################################################################

# Example configuration variables you can define outside or inline here:
PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent

# Add parent to path and change to project root
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Import configuration
from config import main_project_path, clean_level, sample_rate_fmri, intermediate_sample_rate, bandpass_lim, filter_order, transition_width

# Path to a CSV file containing the subjects/runs metadata (e.g., subject ID, run ID).
# This file should have at least two columns: "subject" and "run".
META_DATAFRAME_PATH = PROJECT_ROOT / "dataframes" / "egg_brain_meta_data_v2.csv"

# Path templates for motion confounds and for EGG data:
# Motion files are in: BIDS_data/sub_motion_files/sub-{subject}_dfile.r0{run}.1D
MOTION_FILE_TEMPLATE = os.path.join(
    main_project_path,
    "BIDS_data",
    "sub_motion_files",
    "sub-{sub}_dfile.r0{run}.1D"
)

# EGG data (gastric signal) is stored as .npy in derivatives/brain_gast/
EGG_FILE_TEMPLATE = os.path.join(
    main_project_path,
    "derivatives",
    "brain_gast",
    "{sub}",
    "{sub}{run}",
    "gast_data_{sub}_run{run}{clean_level}.npy"
)

GASTRIC_FREQ_TEMPLATE = os.path.join(
    main_project_path,
    "derivatives",
    "brain_gast",
    "{sub}",
    "{sub}{run}",
    "max_freq{sub}_run{run}{clean_level}.npy"
)

# Some bandpass filter parameters for the EGG band:
SAMPLE_RATE_FMRI = sample_rate_fmri          # The fMRI sampling rate after TR-based downsampling (typical TR=2s => 0.5 Hz).
BANDPASS_LIM = bandpass_lim                   # +/- around the subject-specific peak frequency
FILTER_ORDER = filter_order                   # For filter length (number of cycles)
TRANSITION_WIDTH = transition_width           # Adjust if needed

# If you had originally recorded the EGG at a higher sampling rate and then
# resampled it to an intermediate (like 10 Hz), define that here:
EGG_INTERMEDIATE_SFREQ = intermediate_sample_rate  # e.g., from 5000 Hz down to 10 Hz for EGG.

# Output CSV paths:
OUTPUT_CORREL_PATH = PROJECT_ROOT / "dataframes" / "correl_egg_w_motion.csv"
OUTPUT_PLV_PATH = PROJECT_ROOT / "dataframes" / "plvs_egg_w_motion.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "dataframes" / "motion_summary.csv"

##############################################################################
# Helper Functions                                                           #
##############################################################################

def bp_filter_confounds(df, gastric_peak, sample_rate=SAMPLE_RATE_FMRI,
                        bandpass_lim=BANDPASS_LIM, filter_order=FILTER_ORDER,
                        transition_width=TRANSITION_WIDTH, verbose=None):
    """
    Bandpass-filter each column in df around the subject-specific gastric_peak.
    1. Builds FIR filter length from peak frequency and applies MNE's filter_data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column is a confound time-series at the fMRI sampling rate.
    gastric_peak : float
        The subject-specific peak frequency (in Hz) of the EGG signal.
    sample_rate : float
        Sampling rate of the fMRI time-series in Hz.
    bandpass_lim : float
        Half-width around the gastric peak for the bandpass filter (± bandpass_lim).
    filter_order : int
        Factor to multiply the inverse of the low_cutoff frequency to set filter length.
    transition_width : float
        Transition bandwidth used by MNE filter (as fraction).
    verbose : bool or None
        Verbosity level for MNE filter_data.

    Returns
    -------
    pd.DataFrame
        Filtered confounds with the same shape/columns/index as input df.
    """
    l_freq = gastric_peak - bandpass_lim
    h_freq = gastric_peak + bandpass_lim
    # Calculate filter length in samples
    filter_length = int(filter_order * np.floor(sample_rate / (gastric_peak - bandpass_lim)))

    confound_filtered = filter_data(
        data=df.values.T,
        sfreq=sample_rate,
        l_freq=l_freq,
        h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=transition_width * (gastric_peak - bandpass_lim),
        h_trans_bandwidth=transition_width * (gastric_peak + bandpass_lim),
        n_jobs=1,  # set to number of cores if you want parallel
        method='fir',
        phase='zero-double',
        fir_window='hamming',
        fir_design='firwin2',
        verbose=verbose
    )
    return pd.DataFrame(confound_filtered.T, columns=df.columns, index=df.index)


def calc_plv(signal_a, signal_b, null_median=False):
    """
    Compute the Phase Locking Value (PLV) between two signals of equal length.
    Optionally compute a null distribution by circularly shifting signal_a.

    2. Uses analytic signal (Hilbert) to extract instantaneous phase.

    Parameters
    ----------
    signal_a : 1D np.array
    signal_b : 1D np.array
        Both signals must have the same length.
    null_median : bool
        If True, also compute and return the median PLV from a null distribution
        generated by circularly shifting `signal_a`.

    Returns
    -------
    float or (float, float)
        If null_median=False, returns the empirical PLV.
        If null_median=True, returns a tuple: (empirical_plv, median_null_plv).
    """
    assert len(signal_a) == len(signal_b), "Signals must be the same length."

    # Get instantaneous phase of each signal
    a_phase = np.angle(hilbert(signal_a))
    b_phase = np.angle(hilbert(signal_b))

    # Compute PLV
    plv = np.abs(np.mean(np.exp(1j * (a_phase - b_phase))))

    # Optionally compute the distribution of PLVs for shifted versions of `signal_a`
    if null_median:
        num_shifts = len(signal_a) - 1
        plvs_perm = np.zeros(num_shifts)
        for shift_idx in range(num_shifts):
            rolled = np.roll(signal_a, shift_idx + 1)
            rolled_phase = np.angle(hilbert(rolled))
            plvs_perm[shift_idx] = np.abs(
                np.mean(np.exp(1j * (rolled_phase - b_phase)))
            )
        median_null = np.median(plvs_perm)
        return plv, median_null

    return plv


def get_motion_column_names():
    """
    Define which motion confound columns to use from the .1D motion files.
    The .1D files typically contain 6 columns: roll, pitch, yaw, dS, dL, dP
    (or trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
    """
    return ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


def scatter_plot(x, y, x_label, y_label, save_path=None, test='pearson'):
    """
    Generate a quick scatter plot of x vs y with correlation annotation.
    4. Simple visual QC for relationships between gastric signal and confounds.
    """
    if test == 'pearson':
        r, p = stats.pearsonr(x, y)
        stat_name = 'r'
    else:
        # Spearman
        r, p = stats.spearmanr(x, y)
        stat_name = 'rho'

    annot_text = f"{stat_name}={r:.3f}, p={p:.3f}"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    annot = ax.annotate(annot_text, xy=(.05, .9), xycoords=ax.transAxes, size=12)
    annot.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='k'))

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def confound_summary(df=None, method='mean_abs'):
    """
    Summarize confounds for a quick measure of their amplitude or variability.
    5. Useful to generate a compact per-run summary for later QC or regression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of confounds.
    method : str
        'mean_abs' => mean absolute value
        'mean_abs_diff' => mean absolute derivative

    Returns
    -------
    dict
        Dictionary of summary measures keyed by confound name (e.g., "trans_x_mean_abs").
    """
    measures = ['trans_x', 'trans_y', 'trans_z',
                'rot_x', 'rot_y', 'rot_z']

    out_dict = {}
    for m in measures:
        if m not in df.columns:
            continue
        if method == 'mean_abs':
            stat = np.mean(np.abs(df[m].values))
            out_dict[f"{m}_{method}"] = stat
        elif method == 'mean_abs_diff':
            stat = np.mean(np.abs(np.diff(df[m].values)))
            out_dict[f"{m}_{method}"] = stat
    return out_dict

##############################################################################
# Main Analysis                                                              #
##############################################################################

def main():
    """
    Main function to compute EGG–Motion Confounds PLV for each subject/run
    and save results to CSV files.
    """
    # Load the subject-run metadata
    record_meta_pd = pd.read_csv(META_DATAFRAME_PATH)

    # Apply filtering if strict cardiac cleaning mode is enabled
    if clean_level == 'strict_gs_cardiac':
        record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_exclude'] == False, :]
        record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]

    # Prepare containers for final output
    subjects_runs = list(zip(record_meta_pd['subject'], record_meta_pd['run']))

    # We will dynamically build the columns for these DataFrames once we know what confounds appear
    confound_correl = None
    confound_plvs = None
    confounds_summary_list = []

    # Iterate over each subject-run
    for (subject_name, run) in subjects_runs:
        try:
            # 1. Load the motion .1D file
            motion_path = MOTION_FILE_TEMPLATE.format(sub=subject_name, run=run)
            if not os.path.isfile(motion_path):
                print(f"Motion file not found: {motion_path}")
                continue

            # Load motion data (6 columns: typically trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
            motion_data = np.loadtxt(motion_path)
            motion_cols = get_motion_column_names()
            df_confound = pd.DataFrame(motion_data, columns=motion_cols)

            # 2. Load the gastric EGG signal and peak frequency
            egg_file = EGG_FILE_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)
            freq_file = GASTRIC_FREQ_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)

            if not os.path.isfile(egg_file) or not os.path.isfile(freq_file):
                print(f"EGG or frequency file missing for subject {subject_name}, run {run}")
                continue

            gastric_signal = np.load(egg_file)
            gastric_peak = float(np.load(freq_file).flatten()[0])  # e.g., 0.046 Hz

            # 3. Resample EGG to the fMRI sampling rate (from EGG_INTERMEDIATE_SFREQ to SAMPLE_RATE_FMRI)
            n_points_fmri = int( (len(gastric_signal) / EGG_INTERMEDIATE_SFREQ) * SAMPLE_RATE_FMRI )
            if n_points_fmri < 10:
                print(f"Warning: computed fMRI time series length is too short for {subject_name} run {run}.")
                continue
            gastric_signal_resampled = resample(gastric_signal, n_points_fmri)

            # Match motion data length to gastric signal length
            min_length = min(len(gastric_signal_resampled), len(df_confound))
            gastric_signal_resampled = gastric_signal_resampled[:min_length]
            df_confound = df_confound.iloc[:min_length]

            # 4. Bandpass-filter the confounds around the gastric frequency
            df_confound_filt = bp_filter_confounds(df_confound[motion_cols],
                                                   gastric_peak=gastric_peak,
                                                   sample_rate=SAMPLE_RATE_FMRI,
                                                   bandpass_lim=BANDPASS_LIM,
                                                   filter_order=FILTER_ORDER,
                                                   transition_width=TRANSITION_WIDTH,
                                                   verbose=False)

            # 5. For each confound, compute correlation (r, p) and PLV (with null)
            if confound_correl is None:
                # Build DataFrame columns
                correl_cols = ['subject', 'run'] + [c for c in motion_cols] + [c + "_p" for c in motion_cols]
                confound_correl = pd.DataFrame(columns=correl_cols)

                plv_cols = ['subject', 'run'] + [c for c in motion_cols] + [c + "_null_median" for c in motion_cols]
                confound_plvs = pd.DataFrame(columns=plv_cols)

            # Add row if subject-run not yet there
            row_index = len(confound_correl)
            confound_correl.loc[row_index, 'subject'] = subject_name
            confound_correl.loc[row_index, 'run'] = run

            confound_plvs.loc[row_index, 'subject'] = subject_name
            confound_plvs.loc[row_index, 'run'] = run

            # Summaries for motion
            summary_dict = confound_summary(df_confound_filt, method='mean_abs')
            summary_dict.update({'subject': subject_name, 'run': run})
            confounds_summary_list.append(summary_dict)

            # Actually compute
            for confound in motion_cols:
                sig_confound = df_confound_filt[confound].values
                # corr
                r_val, p_val = stats.pearsonr(gastric_signal_resampled, sig_confound)
                confound_correl.loc[row_index, confound] = r_val
                confound_correl.loc[row_index, confound + "_p"] = p_val

                # plv
                plv, plv_null_median = calc_plv(gastric_signal_resampled, sig_confound, null_median=True)
                confound_plvs.loc[row_index, confound] = plv
                confound_plvs.loc[row_index, confound + "_null_median"] = plv_null_median

            print(f"Processed subject {subject_name}, run {run} successfully.")
        except Exception as e:
            print(f"Error processing subject {subject_name}, run {run}: {e}")

    # 6. Save to CSV
    if confound_correl is not None:
        confound_correl.to_csv(OUTPUT_CORREL_PATH, index=False)
        confound_plvs.to_csv(OUTPUT_PLV_PATH, index=False)

    if len(confounds_summary_list) > 0:
        summary_df = pd.DataFrame(confounds_summary_list)
        summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("Done. Results saved:\n"
          f"  Correlations => {OUTPUT_CORREL_PATH}\n"
          f"  PLV =>         {OUTPUT_PLV_PATH}\n"
          f"  Summary =>     {OUTPUT_SUMMARY_PATH}\n")


if __name__ == "__main__":
    main()