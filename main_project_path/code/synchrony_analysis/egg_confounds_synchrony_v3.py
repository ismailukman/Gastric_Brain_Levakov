import os
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import resample, hilbert
from scipy.stats import false_discovery_control
from mne.filter import filter_data
import matplotlib.pyplot as plt

##############################################################################
# Configuration                                                              #
##############################################################################

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from config import main_project_path, clean_level, sample_rate_fmri, intermediate_sample_rate, bandpass_lim, filter_order, transition_width

META_DATAFRAME_PATH = PROJECT_ROOT / "dataframes" / "egg_brain_meta_data_v2.csv"

MOTION_FILE_TEMPLATE = os.path.join(
    main_project_path,
    "BIDS_data",
    "sub_motion_files",
    "sub-{sub}_dfile.r0{run}.1D"
)

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

SAMPLE_RATE_FMRI = sample_rate_fmri
BANDPASS_LIM = bandpass_lim
FILTER_ORDER = filter_order
TRANSITION_WIDTH = transition_width
EGG_INTERMEDIATE_SFREQ = intermediate_sample_rate

# Output paths
OUTPUT_PLV_PATH = PROJECT_ROOT / "dataframes" / "plvs_egg_w_motion_v3.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "dataframes" / "motion_summary_v3.csv"

##############################################################################
# Helper Functions                                                           #
##############################################################################

def bp_filter_confounds(df, gastric_peak, sample_rate=SAMPLE_RATE_FMRI,
                        bandpass_lim=BANDPASS_LIM, filter_order=FILTER_ORDER,
                        transition_width=TRANSITION_WIDTH, verbose=None):
    """
    Bandpass-filter each column in df around the subject-specific gastric_peak.
    """
    l_freq = gastric_peak - bandpass_lim
    h_freq = gastric_peak + bandpass_lim
    filter_length = int(filter_order * np.floor(sample_rate / (gastric_peak - bandpass_lim)))

    confound_filtered = filter_data(
        data=df.values.T,
        sfreq=sample_rate,
        l_freq=l_freq,
        h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=transition_width * (gastric_peak - bandpass_lim),
        h_trans_bandwidth=transition_width * (gastric_peak + bandpass_lim),
        n_jobs=1,
        method='fir',
        phase='zero-double',
        fir_window='hamming',
        fir_design='firwin2',
        verbose=verbose
    )
    return pd.DataFrame(confound_filtered.T, columns=df.columns, index=df.index)


def calc_plv(signal_a, signal_b):
    """
    Compute the Phase Locking Value (PLV) between two signals of equal length.
    Uses analytic signal (Hilbert) to extract instantaneous phase.

    Parameters
    ----------
    signal_a : 1D np.array
    signal_b : 1D np.array
        Both signals must have the same length.

    Returns
    -------
    float
        The empirical PLV between signal_a and signal_b.
    """
    assert len(signal_a) == len(signal_b), "Signals must be the same length."

    # Get instantaneous phase of each signal
    a_phase = np.angle(hilbert(signal_a))
    b_phase = np.angle(hilbert(signal_b))

    # Compute PLV
    plv = np.abs(np.mean(np.exp(1j * (a_phase - b_phase))))

    return plv


def get_motion_column_names():
    """
    Define which motion confound columns to use from the .1D motion files.
    6 parameters: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
    """
    return ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


def confound_summary(df=None, method='mean_abs'):
    """
    Summarize confounds for quick measure of their amplitude.
    """
    measures = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

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
    Main function to compute EGG-Motion Synchrony using PLV with mismatch null distribution.

    CHANGES IN V3:
    1. Removed correlation analysis - focus on PLV synchronization
    2. Replaced circular permutation with mismatch data null distribution
    3. reviewed p-values with FDR and Bonferroni correction for multiple comparisons
    4. Output includes empirical PLV, p-value, FDR-corrected p, and Bonferroni-corrected p
    """
    # Load the subject-run metadata
    record_meta_pd = pd.read_csv(META_DATAFRAME_PATH)

    # Apply filtering if strict cardiac cleaning mode is enabled
    if clean_level == 'strict_gs_cardiac':
        record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_exclude'] == False, :]
        record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]

    subjects_runs = list(zip(record_meta_pd['subject'], record_meta_pd['run']))
    motion_cols = get_motion_column_names()

    print(f"Processing {len(subjects_runs)} subject-run pairs...")
    print("Loading all gastric signals for mismatch null distribution...")

    # ========================================================================
    # STEP 1: Load all gastric signals and motion data
    # ========================================================================
    all_data = {}

    for (subject_name, run) in subjects_runs:
        try:
            # Load motion data
            motion_path = MOTION_FILE_TEMPLATE.format(sub=subject_name, run=run)
            if not os.path.isfile(motion_path):
                print(f"⚠ Motion file not found: {motion_path}")
                continue

            motion_data = np.loadtxt(motion_path)
            df_confound = pd.DataFrame(motion_data, columns=motion_cols)

            # Load gastric signal and peak frequency
            egg_file = EGG_FILE_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)
            freq_file = GASTRIC_FREQ_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)

            if not os.path.isfile(egg_file) or not os.path.isfile(freq_file):
                print(f"⚠ EGG or frequency file missing for {subject_name} run {run}")
                continue

            gastric_signal = np.load(egg_file)
            gastric_peak = float(np.load(freq_file).flatten()[0])

            # Resample EGG to fMRI sampling rate
            n_points_fmri = int((len(gastric_signal) / EGG_INTERMEDIATE_SFREQ) * SAMPLE_RATE_FMRI)
            if n_points_fmri < 10:
                print(f"⚠ Computed fMRI time series too short for {subject_name} run {run}")
                continue

            gastric_signal_resampled = resample(gastric_signal, n_points_fmri)

            # Match lengths
            min_length = min(len(gastric_signal_resampled), len(df_confound))
            gastric_signal_resampled = gastric_signal_resampled[:min_length]
            df_confound = df_confound.iloc[:min_length]

            # Bandpass-filter motion around gastric frequency
            df_confound_filt = bp_filter_confounds(
                df_confound[motion_cols],
                gastric_peak=gastric_peak,
                sample_rate=SAMPLE_RATE_FMRI,
                bandpass_lim=BANDPASS_LIM,
                filter_order=FILTER_ORDER,
                transition_width=TRANSITION_WIDTH,
                verbose=False
            )

            # Store
            all_data[(subject_name, run)] = {
                'gastric': gastric_signal_resampled,
                'gastric_peak': gastric_peak,
                'motion_filtered': df_confound_filt,
                'motion_raw': df_confound
            }

            print(f"✓ Loaded {subject_name} run {run}")

        except Exception as e:
            print(f"✗ Error loading {subject_name} run {run}: {e}")

    if len(all_data) == 0:
        print("No valid data found. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(all_data)} subject-run pairs")

    # ========================================================================
    # STEP 2: Compute empirical PLV and mismatch null distribution
    # ========================================================================
    print("\nComputing PLV with mismatch null distribution...")

    results = []
    summary_list = []

    for idx, ((subj, run), data) in enumerate(all_data.items()):
        print(f"\nProcessing {subj} run {run} ({idx+1}/{len(all_data)})...")

        gastric_current = data['gastric']
        motion_filt = data['motion_filtered']
        motion_raw = data['motion_raw']

        # Summary statistics
        summary_dict = confound_summary(motion_filt, method='mean_abs')
        summary_dict.update({'subject': subj, 'run': run})
        summary_list.append(summary_dict)

        # For each motion parameter
        for motion_param in motion_cols:
            motion_signal = motion_filt[motion_param].values

            # Compute empirical PLV
            plv_empirical = calc_plv(gastric_current, motion_signal)

            # Compute mismatch null distribution
            # Use gastric signals from all OTHER subject-runs
            null_plvs = []

            for (other_subj, other_run), other_data in all_data.items():
                # Skip self
                if other_subj == subj and other_run == run:
                    continue

                gastric_mismatch = other_data['gastric']

                # Match lengths (use minimum)
                min_len = min(len(motion_signal), len(gastric_mismatch))

                # Compute PLV with mismatched gastric signal
                plv_null = calc_plv(gastric_mismatch[:min_len], motion_signal[:min_len])
                null_plvs.append(plv_null)

            # Compute p-value: proportion of null >= empirical
            null_plvs = np.array(null_plvs)
            p_value = np.mean(null_plvs >= plv_empirical)

            # Store results
            results.append({
                'subject': subj,
                'run': run,
                'motion_param': motion_param,
                'plv_empirical': plv_empirical,
                'plv_null_median': np.median(null_plvs),
                'plv_null_mean': np.mean(null_plvs),
                'plv_null_std': np.std(null_plvs),
                'p_value': p_value,
                'n_null_samples': len(null_plvs)
            })

            print(f"  {motion_param}: PLV={plv_empirical:.4f}, p={p_value:.4f}")

    # ========================================================================
    # STEP 3: Multiple comparison correction
    # ========================================================================
    print("\nApplying multiple comparison correction...")

    results_df = pd.DataFrame(results)

    # Extract p-values
    p_values = results_df['p_value'].values

    # FDR correction (Benjamini-Hochberg)
    p_fdr = false_discovery_control(p_values, method='bh')

    # Bonferroni correction
    p_bonferroni = np.minimum(p_values * len(p_values), 1.0)

    # Add corrected p-values to dataframe
    results_df['p_fdr'] = p_fdr
    results_df['p_bonferroni'] = p_bonferroni

    # Add significance flags
    results_df['sig_uncorrected'] = results_df['p_value'] < 0.05
    results_df['sig_fdr'] = results_df['p_fdr'] < 0.05
    results_df['sig_bonferroni'] = results_df['p_bonferroni'] < 0.05

    # ========================================================================
    # STEP 4: Save results
    # ========================================================================
    print("\nSaving results...")

    # Reorder columns for clarity
    output_cols = [
        'subject', 'run', 'motion_param',
        'plv_empirical', 'plv_null_median', 'plv_null_mean', 'plv_null_std',
        'p_value', 'p_fdr', 'p_bonferroni',
        'sig_uncorrected', 'sig_fdr', 'sig_bonferroni',
        'n_null_samples'
    ]

    results_df = results_df[output_cols]
    results_df.to_csv(OUTPUT_PLV_PATH, index=False)

    # Save motion summary
    if len(summary_list) > 0:
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    # ========================================================================
    # STEP 5: Print summary statistics
    # ========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nTotal tests performed: {len(results_df)}")
    print(f"  Subject-runs: {len(all_data)}")
    print(f"  Motion parameters per run: {len(motion_cols)}")
    print(f"  Total comparisons: {len(all_data)} × {len(motion_cols)} = {len(results_df)}")

    print(f"\nSignificant findings:")
    print(f"  Uncorrected (p < 0.05): {results_df['sig_uncorrected'].sum()} ({100*results_df['sig_uncorrected'].mean():.1f}%)")
    print(f"  FDR-corrected (q < 0.05): {results_df['sig_fdr'].sum()} ({100*results_df['sig_fdr'].mean():.1f}%)")
    print(f"  Bonferroni-corrected (p < 0.05): {results_df['sig_bonferroni'].sum()} ({100*results_df['sig_bonferroni'].mean():.1f}%)")

    print(f"\nPLV statistics:")
    print(f"  Empirical PLV range: [{results_df['plv_empirical'].min():.4f}, {results_df['plv_empirical'].max():.4f}]")
    print(f"  Empirical PLV mean: {results_df['plv_empirical'].mean():.4f} ± {results_df['plv_empirical'].std():.4f}")
    print(f"  Null PLV mean: {results_df['plv_null_mean'].mean():.4f} ± {results_df['plv_null_mean'].std():.4f}")

    # Show most significant findings
    print(f"\nTop 10 most significant findings (by FDR-corrected p-value):")
    top_findings = results_df.nsmallest(10, 'p_fdr')[['subject', 'run', 'motion_param', 'plv_empirical', 'p_value', 'p_fdr']]
    print(top_findings.to_string(index=False))

    print("\n" + "="*70)
    print(f"\nResults saved:")
    print(f"  PLV with statistics => {OUTPUT_PLV_PATH}")
    print(f"  Motion summary      => {OUTPUT_SUMMARY_PATH}")
    print("\nDone!")


if __name__ == "__main__":
    main()
