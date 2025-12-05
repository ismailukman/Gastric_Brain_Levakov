import os
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import resample, hilbert
from scipy.stats import false_discovery_control, mannwhitneyu
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

# Output paths - V4 filenames
OUTPUT_PLV_PATH = PROJECT_ROOT / "dataframes" / "plvs_egg_w_motion_v4.csv"
OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "dataframes" / "motion_summary_v4.csv"
OUTPUT_POPULATION_PATH = PROJECT_ROOT / "dataframes" / "population_level_v4.csv"

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

    VERSION 4 CHANGES (from v3):
    - CORRECTED null distribution: Excludes ALL runs from same subject (not just current run)
    - This ensures true independence in null distribution
    - Null sample size: 25-26 per test (varies by subject) instead of 27
    - More conservative and methodologically sound approach

    CHANGES IN V3 (from v2):
    1. Removed correlation analysis - focus on PLV synchronization
    2. Replaced circular permutation with mismatch data null distribution
    3. Added p-values with FDR and Bonferroni correction for multiple comparisons
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
    print("V4: Using corrected null distribution (excludes ALL same-subject runs)")

    # ========================================================================
    # STEP 1: Load all gastric signals and motion data
    # ========================================================================
    all_data = {}

    for (subject_name, run) in subjects_runs:
        try:
            # Load motion data
            motion_path = MOTION_FILE_TEMPLATE.format(sub=subject_name, run=run)
            if not os.path.isfile(motion_path):
                print(f"Motion file not found: {motion_path}")
                continue

            motion_data = np.loadtxt(motion_path)
            df_confound = pd.DataFrame(motion_data, columns=motion_cols)

            # Load gastric signal and peak frequency
            egg_file = EGG_FILE_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)
            freq_file = GASTRIC_FREQ_TEMPLATE.format(sub=subject_name, run=run, clean_level=clean_level)

            if not os.path.isfile(egg_file) or not os.path.isfile(freq_file):
                print(f"EGG or frequency file missing for {subject_name} run {run}")
                continue

            gastric_signal = np.load(egg_file)
            gastric_peak = float(np.load(freq_file).flatten()[0])

            # Resample EGG to fMRI sampling rate
            n_points_fmri = int((len(gastric_signal) / EGG_INTERMEDIATE_SFREQ) * SAMPLE_RATE_FMRI)
            if n_points_fmri < 10:
                print(f"Computed fMRI time series too short for {subject_name} run {run}")
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
            # V4 CORRECTION: Use gastric signals from OTHER SUBJECTS only (not other runs from same subject)
            null_plvs = []

            for (other_subj, other_run), other_data in all_data.items():
                # Skip ALL runs from same subject (ensure independence)
                if other_subj == subj:
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

            print(f"  {motion_param}: PLV={plv_empirical:.4f}, p={p_value:.4f} (n_null={len(null_plvs)})")

    # ========================================================================
    # STEP 2b: Population-level test per motion parameter
    # ========================================================================
    print("\n" + "="*70)
    print("POPULATION-LEVEL ANALYSIS")
    print("="*70)
    print("\nTesting if empirical PLV > null PLV across all subjects...")
    print("(One test per motion parameter)\n")

    population_results = []
    results_df = pd.DataFrame(results)

    for motion_param in motion_cols:
        # Get all empirical and null PLVs for this motion parameter
        param_data = results_df[results_df['motion_param'] == motion_param]

        empirical_plvs = param_data['plv_empirical'].values

        # For null distribution, we need ALL null PLVs from all runs for this parameter
        # V4 CORRECTION: Each run has 25-26 null samples (excludes all same-subject runs)
        all_null_plvs = []
        for idx, ((subj, run), data) in enumerate(all_data.items()):
            motion_signal = data['motion_filtered'][motion_param].values

            # Collect null PLVs (exclude ALL same-subject runs for independence)
            for (other_subj, other_run), other_data in all_data.items():
                if other_subj == subj:
                    continue
                gastric_mismatch = other_data['gastric']
                min_len = min(len(motion_signal), len(gastric_mismatch))
                plv_null = calc_plv(gastric_mismatch[:min_len], motion_signal[:min_len])
                all_null_plvs.append(plv_null)

        all_null_plvs = np.array(all_null_plvs)

        # Perform Mann-Whitney U test (one-sided: empirical > null)
        # This is a non-parametric test comparing two distributions
        statistic, p_value_mw = mannwhitneyu(
            empirical_plvs,
            all_null_plvs,
            alternative='greater'
        )

        # Also compute effect size (difference in means)
        effect_size = np.mean(empirical_plvs) - np.mean(all_null_plvs)

        population_results.append({
            'motion_param': motion_param,
            'n_empirical': len(empirical_plvs),
            'n_null': len(all_null_plvs),
            'mean_plv_empirical': np.mean(empirical_plvs),
            'std_plv_empirical': np.std(empirical_plvs),
            'mean_plv_null': np.mean(all_null_plvs),
            'std_plv_null': np.std(all_null_plvs),
            'effect_size': effect_size,
            'mann_whitney_u': statistic,
            'p_value': p_value_mw
        })

        print(f"{motion_param}:")
        print(f"  Empirical: {np.mean(empirical_plvs):.4f} ± {np.std(empirical_plvs):.4f} (n={len(empirical_plvs)})")
        print(f"  Null:      {np.mean(all_null_plvs):.4f} ± {np.std(all_null_plvs):.4f} (n={len(all_null_plvs)})")
        print(f"  Effect:    {effect_size:.4f}")
        print(f"  p-value:   {p_value_mw:.6f} {'***' if p_value_mw < 0.001 else '**' if p_value_mw < 0.01 else '*' if p_value_mw < 0.05 else 'ns'}\n")

    population_df = pd.DataFrame(population_results)

    # Apply multiple comparison correction for the 6 tests
    pop_p_values = population_df['p_value'].values

    # FDR correction (Benjamini-Hochberg)
    pop_p_fdr = false_discovery_control(pop_p_values, method='bh')

    # Bonferroni correction
    pop_p_bonferroni = np.minimum(pop_p_values * len(pop_p_values), 1.0)

    population_df['p_fdr'] = pop_p_fdr
    population_df['p_bonferroni'] = pop_p_bonferroni
    population_df['sig_uncorrected'] = population_df['p_value'] < 0.05
    population_df['sig_fdr'] = population_df['p_fdr'] < 0.05
    population_df['sig_bonferroni'] = population_df['p_bonferroni'] < 0.05

    print("\nMultiple comparison correction:")
    print("  FDR (Benjamini-Hochberg) for 6 tests:")
    for idx, row in population_df.iterrows():
        sig_marker = '***' if row['p_fdr'] < 0.001 else '**' if row['p_fdr'] < 0.01 else '*' if row['p_fdr'] < 0.05 else 'ns'
        print(f"    {row['motion_param']}: q = {row['p_fdr']:.6f} {sig_marker}")

    print("\n  Bonferroni for 6 tests:")
    for idx, row in population_df.iterrows():
        sig_marker = '***' if row['p_bonferroni'] < 0.001 else '**' if row['p_bonferroni'] < 0.01 else '*' if row['p_bonferroni'] < 0.05 else 'ns'
        print(f"    {row['motion_param']}: p_bonf = {row['p_bonferroni']:.6f} {sig_marker}")

    # ========================================================================
    # STEP 3: Multiple comparison correction (individual tests)
    # ========================================================================
    print("\n" + "="*70)
    print("INDIVIDUAL-LEVEL ANALYSIS")
    print("="*70)
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

    # Save population-level results
    population_df.to_csv(OUTPUT_POPULATION_PATH, index=False)

    # ========================================================================
    # STEP 5: Print summary statistics
    # ========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY (VERSION 4)")
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

    print(f"\nNull distribution statistics (V4 correction):")
    print(f"  Null sample size range: [{results_df['n_null_samples'].min()}, {results_df['n_null_samples'].max()}]")
    print(f"  Mean null samples per test: {results_df['n_null_samples'].mean():.1f}")

    # Show most significant findings
    print(f"\nTop 10 most significant findings (by FDR-corrected p-value):")
    top_findings = results_df.nsmallest(10, 'p_fdr')[['subject', 'run', 'motion_param', 'plv_empirical', 'p_value', 'p_fdr']]
    print(top_findings.to_string(index=False))

    print("\n" + "="*70)
    print("POPULATION-LEVEL RESULTS SUMMARY")
    print("="*70)

    print(f"\nSignificance by correction method:")
    print(f"  Uncorrected (p < 0.05): {population_df['sig_uncorrected'].sum()}/6")
    print(f"  FDR-corrected (q < 0.05): {population_df['sig_fdr'].sum()}/6")
    print(f"  Bonferroni-corrected (p < 0.05): {population_df['sig_bonferroni'].sum()}/6")

    print(f"\nMotion parameters with Bonferroni-significant effects:")
    sig_pop = population_df[population_df['sig_bonferroni']]
    if len(sig_pop) > 0:
        for idx, row in sig_pop.iterrows():
            print(f"  {row['motion_param']}: effect={row['effect_size']:.4f}, q={row['p_fdr']:.6f}, p_bonf={row['p_bonferroni']:.6f}")
    else:
        print("  None")

    print(f"\nMotion parameters with FDR-significant effects (not Bonferroni):")
    sig_fdr_only = population_df[(population_df['sig_fdr']) & (~population_df['sig_bonferroni'])]
    if len(sig_fdr_only) > 0:
        for idx, row in sig_fdr_only.iterrows():
            print(f"  {row['motion_param']}: effect={row['effect_size']:.4f}, q={row['p_fdr']:.6f}, p_bonf={row['p_bonferroni']:.6f}")
    else:
        print("  None")

    print("\n" + "="*70)
    print(f"\nResults saved:")
    print(f"  Individual-level PLV => {OUTPUT_PLV_PATH}")
    print(f"  Population-level     => {OUTPUT_POPULATION_PATH}")
    print(f"  Motion summary       => {OUTPUT_SUMMARY_PATH}")
    print("\nDone! (Version 4 - Corrected null distribution)")


if __name__ == "__main__":
    main()
