"""
Extract motion parameters from AFNI preprocessing output.

AFNI typically outputs motion parameters in .1D files. This script can extract
them from AFNI output if you don't already have them in sub_motion_files.

Usage:
    python synchrony_analysis/extract_afni_motion.py <subject>

Example:
    python synchrony_analysis/extract_afni_motion.py AlM
"""

import os
import sys
import argparse
import shutil
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('subject', help='Subject name (e.g., AlM, AE)')
parser.add_argument('--afni-base', default=None, help='Base path to AFNI data')
parser.add_argument('--output-dir', default=None, help='Output directory for motion files')
args = parser.parse_args()

subject_name = args.subject

# Set paths
if args.afni_base:
    afni_base = args.afni_base
else:
    afni_base = '../BIDS_data/soroka'

if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = '../BIDS_data/sub_motion_files'

afni_subject_dir = f'{afni_base}/sub-{subject_name}/anat_func/PreprocessedData/sub-{subject_name}/output_sub-{subject_name}'

print(f'Extracting motion parameters for subject: {subject_name}')
print(f'AFNI directory: {afni_subject_dir}')

# Check if AFNI directory exists
if not os.path.exists(afni_subject_dir):
    print(f'ERROR: AFNI directory not found: {afni_subject_dir}')
    print(f'\nSubject {subject_name} does not appear to have AFNI preprocessing.')
    print('Motion parameters cannot be extracted.')
    sys.exit(1)

# Look for motion parameter files in AFNI output
# AFNI typically creates dfile.rXX.1D files with motion parameters
motion_files = []
for f in os.listdir(afni_subject_dir):
    if 'dfile.r' in f and f.endswith('.1D'):
        motion_files.append(f)

if not motion_files:
    print(f'\nERROR: No motion parameter files (dfile.r*.1D) found in {afni_subject_dir}')
    print('\nSearching for alternative motion files...')

    # Try looking for motion parameters in other locations
    alt_patterns = ['motion', 'dfile', 'volreg']
    for pattern in alt_patterns:
        for f in os.listdir(afni_subject_dir):
            if pattern in f.lower() and f.endswith('.1D'):
                motion_files.append(f)

    if not motion_files:
        print('No motion parameter files found.')
        print('\nTip: Check the AFNI output directory manually:')
        print(f'  ls {afni_subject_dir}/*.1D')
        sys.exit(1)

print(f'\nFound {len(motion_files)} motion parameter files:')
for mf in sorted(motion_files):
    print(f'  - {mf}')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Copy motion files to output directory with standard naming
copied = 0
for motion_file in sorted(motion_files):
    # Extract run number from filename
    # Expected format: dfile.r01.1D or similar
    if '.r' in motion_file:
        run_part = motion_file.split('.r')[1].split('.')[0]
        run_num = int(run_part)

        src_path = os.path.join(afni_subject_dir, motion_file)
        dst_path = os.path.join(output_dir, f'sub-{subject_name}_dfile.r{run_part}.1D')

        # Check if file already exists
        if os.path.exists(dst_path):
            print(f'\n⚠ File already exists: {dst_path}')
            overwrite = input('Overwrite? (y/n): ')
            if overwrite.lower() != 'y':
                print('Skipping...')
                continue

        # Copy file
        shutil.copy2(src_path, dst_path)
        print(f'\n✓ Copied: {motion_file}')
        print(f'  → {dst_path}')

        # Verify file format (should be 6 columns)
        data = np.loadtxt(dst_path)
        print(f'  Shape: {data.shape} (should be [n_timepoints, 6])')

        if data.shape[1] != 6:
            print(f'  ⚠ Warning: Expected 6 columns, got {data.shape[1]}')

        copied += 1

print(f'\n{"="*70}')
if copied > 0:
    print(f'SUCCESS: Extracted {copied} motion parameter files')
    print(f'Output location: {os.path.abspath(output_dir)}')
    print(f'\nYou can now run:')
    print(f'  python synchrony_analysis/check_files_v2.py')
else:
    print('No files were copied.')

print(f'{"="*70}')