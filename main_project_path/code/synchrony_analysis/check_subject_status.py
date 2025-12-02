"""
Check which subjects have complete data and which are missing components.

This script provides a detailed status report for all subjects, showing:
- Which have complete data (in egg_brain_meta_data_v2.csv)
- Which have partial data (AIM, AlM)
- What files are missing for each partial subject

Usage:
    cd code
    python synchrony_analysis/check_subject_status.py
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import main_project_path, clean_level

def check_motion_file(subject, run):
    """Check if motion file exists for subject/run."""
    motion_path = f'{main_project_path}/BIDS_data/sub_motion_files/sub-{subject}_dfile.r0{run}.1D'
    return os.path.exists(motion_path)

def check_gastric_file(subject, run):
    """Check if gastric file exists for subject/run."""
    gastric_path = f'{main_project_path}/derivatives/brain_gast/{subject}/{subject}{run}/gast_data_{subject}_run{run}{clean_level}.npy'
    return os.path.exists(gastric_path)

def check_afni_preprocessing(subject):
    """Check if AFNI preprocessing exists for subject."""
    afni_path = f'{main_project_path}/BIDS_data/soroka/sub-{subject}/anat_func/PreprocessedData'
    return os.path.exists(afni_path)

def main():
    print("="*80)
    print("SUBJECT DATA STATUS REPORT")
    print("="*80)

    # Load the v2 metadata (14 subjects with complete data)
    meta_v2_path = f'{main_project_path}/code/dataframes/egg_brain_meta_data_v2.csv'
    df_v2 = pd.read_csv(meta_v2_path)
    subjects_v2 = df_v2['subject'].unique()

    print(f"\n✓ SUBJECTS WITH COMPLETE DATA ({len(subjects_v2)} subjects):")
    print("-"*80)

    # Group by subject and count runs
    subject_runs = df_v2.groupby('subject')['run'].apply(list).to_dict()

    # Separate AFNI-ready from motion+gastric only
    afni_ready = []
    motion_gastric_only = []

    for subject in sorted(subjects_v2):
        runs = subject_runs[subject]
        has_afni = check_afni_preprocessing(subject)

        if has_afni:
            afni_ready.append(f"{subject} (runs {', '.join(map(str, runs))})")
        else:
            motion_gastric_only.append(f"{subject} (runs {', '.join(map(str, runs))})")

    if afni_ready:
        print(f"\n  AFNI + Motion + Gastric ({len(afni_ready)} subjects):")
        for s in afni_ready:
            print(f"    • {s}")

    if motion_gastric_only:
        print(f"\n  Motion + Gastric only ({len(motion_gastric_only)} subjects):")
        for s in motion_gastric_only:
            print(f"    • {s}")

    # Check special cases: AIM and AlM
    print("\n"+"="*80)
    print("⚠  SUBJECTS WITH PARTIAL DATA")
    print("="*80)

    # AIM - has motion + AFNI but no gastric
    print("\n1. AIM - Has Motion + AFNI, Missing Gastric")
    print("-"*80)
    has_motion_r01 = check_motion_file('AIM', 1)
    has_motion_r02 = check_motion_file('AIM', 2)
    has_afni = check_afni_preprocessing('AIM')
    has_gastric_r01 = check_gastric_file('AIM', 1)
    has_gastric_r02 = check_gastric_file('AIM', 2)

    print(f"  Motion run 1:      {'✓ YES' if has_motion_r01 else '✗ NO'}")
    print(f"  Motion run 2:      {'✓ YES' if has_motion_r02 else '✗ NO'}")
    print(f"  AFNI preprocessing: {'✓ YES' if has_afni else '✗ NO'}")
    print(f"  Gastric run 1:     {'✓ YES' if has_gastric_r01 else '✗ NO'}")
    print(f"  Gastric run 2:     {'✓ YES' if has_gastric_r02 else '✗ NO'}")

    if has_motion_r01 and has_afni and not has_gastric_r01:
        print("\n  → Can be included if you process EGG data for AIM")
        print("     1. Run gastric preprocessing for AIM")
        print("     2. Verify gast_data_AIM_run*.npy files are created")
        print("     3. Add AIM to egg_brain_meta_data_v2.csv")

    # AlM - has gastric but no motion
    print("\n2. AlM - Has Gastric, Missing Motion + AFNI")
    print("-"*80)
    has_motion_r01 = check_motion_file('AlM', 1)
    has_motion_r02 = check_motion_file('AlM', 2)
    has_afni = check_afni_preprocessing('AlM')
    has_gastric_r01 = check_gastric_file('AlM', 1)
    has_gastric_r02 = check_gastric_file('AlM', 2)

    print(f"  Motion run 1:      {'✓ YES' if has_motion_r01 else '✗ NO'}")
    print(f"  Motion run 2:      {'✓ YES' if has_motion_r02 else '✗ NO'}")
    print(f"  AFNI preprocessing: {'✓ YES' if has_afni else '✗ NO'}")
    print(f"  Gastric run 1:     {'✓ YES' if has_gastric_r01 else '✗ NO'}")
    print(f"  Gastric run 2:     {'✓ YES' if has_gastric_r02 else '✗ NO'}")

    if has_gastric_r01 and not has_motion_r01:
        print("\n  → Can be included if you can locate/generate motion parameters")
        print("     Option 1: If you have raw fMRI for AlM")
        print("       1. Run AFNI or fMRIPrep preprocessing")
        print("       2. Extract motion parameters")
        print("       3. Add AlM to egg_brain_meta_data_v2.csv")
        print("\n     Option 2: If AlM was preprocessed elsewhere")
        print("       1. Locate motion parameter files")
        print("       2. Convert to .1D format (6 columns)")
        print("       3. Copy to BIDS_data/sub_motion_files/")
        print("       4. Add AlM to egg_brain_meta_data_v2.csv")

    # Summary
    print("\n"+"="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n  Ready to analyze:  {len(subjects_v2)} subjects")
    print(f"  AFNI + ready:      {len(afni_ready)} subjects (can run full pipeline now)")
    print(f"  Need brain prep:   {len(motion_gastric_only)} subjects (have motion + gastric)")
    print(f"  Partial data:      2 subjects (AIM, AlM)")

    print("\n"+"="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\nTo process AFNI-ready subjects:")
    print("  cd code")
    print("  python synchrony_analysis/prepare_afni_data.py <subject> <run>")
    print("  python synchrony_analysis/signal_slicing_v2.py <subject> <run>")
    print("  python synchrony_analysis/voxel_based_analysis_v2.py <subject> <run>")

    if afni_ready:
        example_subject = afni_ready[0].split(' ')[0]
        print(f"\nExample (first AFNI subject):")
        print(f"  python synchrony_analysis/prepare_afni_data.py {example_subject} 1")

    print("\n"+"="*80)

if __name__ == '__main__':
    main()