"""
Prepare AFNI preprocessed data for synchrony analysis.

This script converts AFNI output to the format required by synchrony_analysis scripts.
It creates:
1. func_filtered_*.npz - brain signal data
2. mask_*.npz - brain masks

Input: AFNI preprocessed data in BIDS_data/soroka/
Output: Files in derivatives/brain_gast/

Usage:
    python synchrony_analysis/prepare_afni_data.py <subject> <run>

Example:
    python synchrony_analysis/prepare_afni_data.py AE 1
"""

import os
import sys
import pathlib
import numpy as np
import nibabel as nib
import argparse
from scipy import signal as sp_signal

# Add parent to path and change to project root
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))

from config import (main_project_path, clean_level, sample_rate_fmri,
                    intermediate_sample_rate, bandpass_lim, filter_order,
                    transition_width, freq_range, brain_fwhm)
from mne.filter import filter_data

# Parse arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='subject_name', help='The subject to process')
parser.add_argument('run', metavar='run_number', help='The run number of the subject')
parser.add_argument('--afni-base', default=None, help='Base path to AFNI data (default: BIDS_data/soroka)')
parser.add_argument('--use-errts', action='store_true', help='Use errts (residuals) instead of pb files')
args = parser.parse_args()

subject_name = args.subject
run = str(args.run)

# Set up paths
if args.afni_base:
    afni_base = args.afni_base
else:
    afni_base = f'{main_project_path}/BIDS_data/soroka'

afni_subject_dir = f'{afni_base}/sub-{subject_name}/anat_func/PreprocessedData/sub-{subject_name}/output_sub-{subject_name}'
derivatives_path = f'{main_project_path}/derivatives/brain_gast/{subject_name}/{subject_name}{run}'
mask_base_path = f'{main_project_path}/derivatives/brain_gast'

# Create output directories if they don't exist
os.makedirs(derivatives_path, exist_ok=True)
os.makedirs(f'{main_project_path}/plots/brain_gast/{subject_name}/{subject_name}{run}', exist_ok=True)

print(f'Processing subject: {subject_name}, run: {run}')
print(f'AFNI data location: {afni_subject_dir}')
print(f'Output location: {derivatives_path}')

# Load gastric signal to get timing info
gastric_file = f'{derivatives_path}/gast_data_{subject_name}_run{run}{clean_level}.npy'
freq_file = f'{derivatives_path}/max_freq{subject_name}_run{run}{clean_level}.npy'

if not os.path.exists(gastric_file):
    raise FileNotFoundError(f"Gastric signal not found: {gastric_file}")
if not os.path.exists(freq_file):
    raise FileNotFoundError(f"Frequency file not found: {freq_file}")

gastric_signal = np.load(gastric_file)
gastric_peak = float(np.load(freq_file).flatten()[0])

print(f'Gastric signal length: {len(gastric_signal)} samples at {intermediate_sample_rate} Hz')
print(f'Gastric peak frequency: {gastric_peak:.4f} Hz')

# Determine which AFNI file to use
if args.use_errts:
    # Use residual timeseries (after regression) - combined runs
    afni_file = f'{afni_subject_dir}/errts.sub-{subject_name}.tproject+tlrc'
    print(f'Using AFNI errts (residuals): {afni_file}')
else:
    # Use individual run preprocessed data (pb files)
    # pb04 = blurred, pb05 = scaled
    afni_file = f'{afni_subject_dir}/pb04.sub-{subject_name}.r0{run}.blur+tlrc'
    if not os.path.exists(afni_file + '.HEAD'):
        afni_file = f'{afni_subject_dir}/pb03.sub-{subject_name}.r0{run}.volreg+tlrc'
    print(f'Using AFNI pb file: {afni_file}')

# Check if AFNI file exists
if not os.path.exists(afni_file + '.HEAD'):
    raise FileNotFoundError(f"AFNI file not found: {afni_file}.HEAD")

# Load AFNI data using nibabel
print('Loading AFNI data...')
img = nib.load(afni_file + '.HEAD')
brain_data = img.get_fdata()

print(f'Brain data shape: {brain_data.shape}')

# If using errts (combined runs), need to extract the specific run
if args.use_errts:
    # errts typically concatenates all runs
    # Need to extract based on run number and expected length
    expected_length = int((len(gastric_signal) / intermediate_sample_rate) * sample_rate_fmri)

    # Assuming 600s (300 TRs) or 900s (450 TRs) per run
    # This needs to be adjusted based on your actual run lengths
    run_num = int(run)
    tr_per_run = brain_data.shape[-1] // 3  # Assuming 3 runs, adjust as needed

    start_tr = (run_num - 1) * tr_per_run
    end_tr = run_num * tr_per_run

    print(f'Extracting run {run} from errts: TRs {start_tr} to {end_tr}')
    brain_data = brain_data[:, :, :, start_tr:end_tr]
    print(f'Extracted brain data shape: {brain_data.shape}')

# Create mask (non-zero voxels)
print('Creating brain mask...')
mean_brain = np.mean(brain_data, axis=-1)
mask = mean_brain > 0

# Reshape brain data to 2D (voxels x time)
print('Reshaping brain data...')
n_voxels = np.prod(brain_data.shape[:3])
n_timepoints = brain_data.shape[3]
brain_data_2d = brain_data.reshape(n_voxels, n_timepoints)

# Extract only masked voxels
masked_voxels = brain_data_2d[mask.flatten(), :]
print(f'Masked brain data shape: {masked_voxels.shape}')

# Bandpass filter brain data in gastric frequency range
print(f'Bandpass filtering brain data around {gastric_peak:.4f} Hz...')
l_freq = gastric_peak - bandpass_lim
h_freq = gastric_peak + bandpass_lim
filter_length = int(filter_order * np.floor(sample_rate_fmri / l_freq))

print(f'Filter: {l_freq:.4f} - {h_freq:.4f} Hz, length: {filter_length} samples')

brain_filtered = filter_data(
    data=masked_voxels,
    sfreq=sample_rate_fmri,
    l_freq=l_freq,
    h_freq=h_freq,
    filter_length=filter_length,
    l_trans_bandwidth=transition_width * l_freq,
    h_trans_bandwidth=transition_width * h_freq,
    n_jobs=1,
    method='fir',
    phase='zero-double',
    fir_window='hamming',
    fir_design='firwin2',
    verbose=False
)

print(f'Filtered brain data shape: {brain_filtered.shape}')

# Match timing with gastric signal
gastric_sr_ratio = sample_rate_fmri / intermediate_sample_rate
expected_brain_length = int(len(gastric_signal) * gastric_sr_ratio)

if brain_filtered.shape[1] != expected_brain_length:
    print(f'Warning: Brain length ({brain_filtered.shape[1]}) != expected ({expected_brain_length})')
    print(f'Truncating to minimum length...')
    min_length = min(brain_filtered.shape[1], expected_brain_length)
    brain_filtered = brain_filtered[:, :min_length]

# Save filtered brain signal
output_brain_file = f'{derivatives_path}/func_filtered_{subject_name}_run{run}{clean_level}.npz'
np.savez_compressed(output_brain_file, brain_signal=brain_filtered)
print(f'Saved filtered brain data to: {output_brain_file}')

# Save mask
output_mask_file = f'{mask_base_path}/mask_{subject_name}_run{run}{clean_level}.npz'
np.savez_compressed(output_mask_file, mask=mask)
print(f'Saved mask to: {output_mask_file}')

# Also save the preprocessed brain image in MNI space for later visualization
# Convert back to NIfTI format
preprocessed_nifti_path = f'{derivatives_path}/{subject_name}_task-rest_run-0{run}_space-MNI_desc-preproc_bold_{clean_level}.nii.gz'
nib.save(img, preprocessed_nifti_path)
print(f'Saved preprocessed NIfTI to: {preprocessed_nifti_path}')

print('\n' + '='*60)
print('SUCCESS! AFNI data prepared for synchrony analysis.')
print('='*60)
print(f'\nGenerated files:')
print(f'  1. {output_brain_file}')
print(f'  2. {output_mask_file}')
print(f'  3. {preprocessed_nifti_path}')
print(f'\nYou can now run:')
print(f'  python synchrony_analysis/signal_slicing_v2.py {subject_name} {run}')
print(f'  python synchrony_analysis/voxel_based_analysis_v2.py {subject_name} {run}')