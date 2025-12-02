# Quick Start Guide - AFNI Data

Simplified guide to run gastric-brain synchrony analysis with AFNI preprocessed data.

**Your Dataset:** 15 subjects (28 runs)
**AFNI-ready:** ALL 15 subjects - 28 runs total
**Status:** ✓ All subjects ready for analysis!

## Prerequisites

1. **Install Python packages:**
```bash
pip install numpy pandas scipy nilearn nibabel matplotlib mne scikit-learn
```

2. **Set up FSL** (required for MNI template):
```bash
# Option 1: Source FSL
source ~/.bash_profile

# Option 2: Set manually
export FSLDIR=/Users/usrname/fsl
export PATH=${FSLDIR}/share/fsl/bin:${PATH}
source ${FSLDIR}/etc/fslconf/fsl.sh
```

3. **Required data:**
   - ✓ Motion files: `BIDS_data/sub_motion_files/*.1D`
   - ✓ Gastric signals: `derivatives/brain_gast/{subject}/{subject}{run}/gast_data_*.npy`
   - ✓ AFNI preprocessed: `BIDS_data/soroka/sub-{subject}/anat_func/PreprocessedData/`


## Step-by-Step: Process Your First Subject

Let's process subject **AE**, run **1** as an example:

### Step 1: Verify Your Files

```bash
cd code

# Check if motion and gastric files exist
python synchrony_analysis/check_files_v2.py
```

Expected output: `Done files pre check.`

### Step 2: Prepare AFNI Data

This converts AFNI output to the format needed for synchrony analysis:

```bash
python synchrony_analysis/prepare_afni_data.py AE 1
```

**What this does:**
- Loads AFNI preprocessed data (pb04 files)
- Bandpass filters in gastric frequency range
- Saves to `derivatives/brain_gast/AE/AE1/func_filtered_*.npz`
- Creates brain mask: `derivatives/brain_gast/mask_*.npz`


### Step 3: Signal Slicing

Align gastric and brain signals temporally:

```bash
python synchrony_analysis/signal_slicing_v2.py AE 1
```

**Expected output:** "signal length: XXX sec"

### Step 4: Voxel-Based PLV Analysis

Compute phase-locking value between gastric and brain signals:

```bash
python synchrony_analysis/voxel_based_analysis_v2.py AE 1
```

**What this creates:**
- PLV maps: `derivatives/brain_gast/AE/AE1/plvs_empirical_*.nii.gz`
- Statistical maps: `plv_p_vals_*.nii.gz`, `plv_delta_*.nii.gz`
- Plots: `plots/brain_gast/AE/AE1/*.png`

**Expected time:** about 15 minutes per run (depends on permutations)

### Step 5 (Optional): Gastric-Motion Synchrony

Analyze relationship between gastric signal and motion:

```bash
python synchrony_analysis/egg_confounds_synchrony_v2.py
```

This processes all subjects in your metadata and creates:
- `dataframes/correl_egg_w_motion.csv`
- `dataframes/plvs_egg_w_motion.csv`

## Process All Your AFNI Subjects

### Option A: Manual (Recommended for First Time)

Process each subject individually:

```bash
cd code

# Subject AE (3 runs)
python synchrony_analysis/prepare_afni_data.py AE 1
python synchrony_analysis/signal_slicing_v2.py AE 1
python synchrony_analysis/voxel_based_analysis_v2.py AE 1

python synchrony_analysis/prepare_afni_data.py AE 2
python synchrony_analysis/signal_slicing_v2.py AE 2
python synchrony_analysis/voxel_based_analysis_v2.py AE 2

# Repeat for all 15 subjects: AE, AIM, AlS, AmK, AnF, AzN, BS, DaH, DoP, EdZ, ElL, ErG, HaM, IdS, LA
```

### Option B: Batch Script

Create a simple batch script:

```bash
cd code

# Create batch processing script
cat > process_afni_subjects.sh << 'EOF'
#!/bin/bash

# All 15 subjects with AFNI preprocessing
SUBJECTS="AE AIM AlS AmK AnF AzN BS DaH DoP EdZ ElL ErG HaM IdS LA"

for SUBJECT in $SUBJECTS; do
    for RUN in 1 2 3; do
        # Check if this subject/run exists in metadata
        if grep -q "^${SUBJECT},${RUN}," dataframes/egg_brain_meta_data_v2.csv; then
            echo "======================================"
            echo "Processing ${SUBJECT} run ${RUN}"
            echo "======================================"

            # Step 1: Prepare AFNI data
            python synchrony_analysis/prepare_afni_data.py $SUBJECT $RUN || continue

            # Step 2: Signal slicing
            python synchrony_analysis/signal_slicing_v2.py $SUBJECT $RUN || continue

            # Step 3: Voxel-based analysis
            python synchrony_analysis/voxel_based_analysis_v2.py $SUBJECT $RUN

            echo "✓ Completed ${SUBJECT} run ${RUN}"
        fi
    done
done

echo ""
echo "All 15 subjects processed!"
EOF

# Make executable and run
chmod +x process_afni_subjects.sh
./process_afni_subjects.sh
```

## Group-Level Analysis

After processing all subjects, run second-level analysis:

```bash
python synchrony_analysis/voxel_based_second_level_v2.py
```

This requires:
- FSL installed (for randomise)
- All individual subjects processed

**Output:**
- Group average maps: `derivatives/brain_gast/plv_*_mean_subjects*.nii.gz`
- Statistical results: `derivatives/brain_gast/fsl_randomize/result_*.nii.gz`

## Expected Output Structure

After processing subject AE run 1, you should have:

```
derivatives/brain_gast/
├── AE/
│   └── AE1/
│       ├── gast_data_AE_run1strict.npy                    (input: gastric signal)
│       ├── max_freqAE_run1strict.npy                      (input: gastric freq)
│       ├── func_filtered_AE_run1strict.npz                (created by prepare_afni_data)
│       ├── gast_data_AE_run1strict_sliced.npy             (created by signal_slicing)
│       ├── func_filtered_AE_run1strict_sliced.npz         (created by signal_slicing)
│       ├── AE_task-rest_run-01_space-MNI_desc-preproc_bold_strict.nii.gz  (from AFNI)
│       ├── plvs_empirical_AE_run1strict.nii.gz            (created by voxel_based)
│       ├── plv_p_vals_AE_run1strict.nii.gz
│       ├── plv_delta_AE_run1strict.nii.gz
│       └── plv_permut_median_AE_run1strict.nii.gz
└── mask_AE_run1strict.npz                                  (created by prepare_afni_data)

plots/brain_gast/AE/AE1/
├── egg_BOLD_sync_example.png
├── empirical_plv_map.png
├── thres95_plv_p_vals_map.png
├── thres95_plv_delta_map.png
├── thres95_plv_permut_median_map.png
└── thres95_plvs_empirical_map.png
```

## Common Issues

### Issue: "AFNI file not found"

**Solution:**
```bash
# Check which pb files exist
ls BIDS_data/soroka/sub-AE/anat_func/PreprocessedData/sub-AE/output_sub-AE/pb*.HEAD

# The script tries pb04, then pb03
# Make sure at least pb03 exists
```

### Issue: "Gastric signal not found"

**Solution:**
You need to preprocess gastric data first. Check if files exist:
```bash
ls derivatives/brain_gast/AE/AE1/*.npy
```

### Issue: "FSL_DIR not set"

This is only needed for voxel_based_analysis. Set it:
```bash
export FSL_DIR=/usr/local/fsl  # Adjust to your FSL installation
```

### Issue: Memory errors

Process subjects one at a time instead of batch processing.

### Issue: "Brain and gastric signals don't match"

This is handled automatically by signal_slicing_v2.py. The warning is informational.

## Visualization

View your results in FSLeyes or AFNI:

```bash
# Using FSLeyes (if installed)
fsleyes derivatives/brain_gast/AE/AE1/plvs_empirical_AE_run1strict.nii.gz

# Using AFNI
afni derivatives/brain_gast/AE/AE1/plvs_empirical_AE_run1strict.nii.gz
```

Or just check the PNG plots:
```bash
open plots/brain_gast/AE/AE1/thres95_plvs_empirical_map.png
```

## Next Steps

1. Process all 15 AFNI subjects (complete dataset)
2. Run group-level analysis with all 28 runs
3. Examine results in `derivatives/brain_gast/`
4. Check plots in `plots/brain_gast/`

## AFNI File Structure & Details

Your AFNI preprocessing creates these files:
```
BIDS_data/soroka/sub-AE/anat_func/PreprocessedData/sub-AE/output_sub-AE/
├── pb04.sub-AE.r01.blur+tlrc.HEAD/.BRIK  (blurred - RECOMMENDED)
├── pb03.sub-AE.r01.volreg+tlrc.HEAD      (motion corrected)
└── errts.sub-AE.tproject+tlrc.HEAD       (residuals)
```

**Which file to use?**
- **pb04 (default):** Smoothed, motion-corrected, MNI space - best for connectivity
- **errts (optional):** Residuals after regression - use `--use-errts` flag

## Need Help?

See detailed documentation:
- [README_v2.md](README_v2.md) - Complete user guide with troubleshooting
- [DEPENDENCIES_v2.txt](DEPENDENCIES_v2.txt) - Full dependency checklist and dataset info
- [START_HERE.txt](START_HERE.txt) - Quick reference commands