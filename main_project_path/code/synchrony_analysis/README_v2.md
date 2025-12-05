# Synchrony Analysis V2 - Complete User Guide

Gastric-Brain Synchrony Analysis Pipeline using AFNI preprocessing and .1D motion files.

## Dataset Overview

**15 subjects, 28 runs** (egg_brain_meta_data_v2.csv)

**ALL SUBJECTS AFNI-READY (15 subjects, 28 runs):**
- AE (3 runs), AIM (2 runs), AlS (2 runs), AmK (2 runs), AnF (2 runs)
- AzN (2 runs), BS (2 runs), DaH (2 runs), DoP (1 run), EdZ (1 run)
- ElL (2 runs), ErG (1 run), HaM (2 runs), IdS (2 runs), LA (2 runs)
- ✓ All subjects have AFNI preprocessing - full pipeline ready!

**Important Note - AIM/AlM Naming:**
AIM and AlM are the **SAME SUBJECT** with inconsistent naming. Resolved by:
- Standardized to "AIM" in metadata
- Created symbolic links: `derivatives/brain_gast/AIM -> AlM`
- Gastric data accessible as both AIM and AlM

## Quick Start

```bash
cd code

# 1. Verify files
python synchrony_analysis/check_files_v2.py

# 2. Process first AFNI subject (AE, run 1)
python synchrony_analysis/prepare_afni_data.py AE 1
python synchrony_analysis/signal_slicing_v2.py AE 1
python synchrony_analysis/voxel_based_analysis_v2.py AE 1

# 3. Check outputs
ls ../derivatives/brain_gast/AE/AE1/*.nii.gz
ls ../plots/brain_gast/AE/AE1/*.png
```

## Prerequisites

### Required Files (Per Subject/Run)

1. **Motion parameters** (.1D files)
   - Location: `BIDS_data/sub_motion_files/sub-{subject}_dfile.r0{run}.1D`
   - Format: 6 columns (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)

2. **Gastric signals** (.npy files)
   - Location: `derivatives/brain_gast/{subject}/{subject}{run}/gast_data_{subject}_run{run}strict.npy`
   - Format: 1D array, sampled at 10 Hz

3. **Gastric frequency** (.npy files)
   - Location: `derivatives/brain_gast/{subject}/{subject}{run}/max_freq{subject}_run{run}strict.npy`
   - Format: Single float value (peak frequency)

4. **AFNI preprocessing** (for 5 subjects)
   - Location: `BIDS_data/soroka/sub-{subject}/anat_func/PreprocessedData/`
   - Available for: AE, AIM, AlS, AmK, AnF

5. **Metadata CSV**
   - Location: `code/dataframes/egg_brain_meta_data_v2.csv`
   - Contains: subject, run, and experimental parameters

### Python Packages

```bash
pip install numpy pandas scipy nilearn nibabel matplotlib mne scikit-learn nipype
```

### External Software (Optional)

- **FSL**: For second-level analysis (randomise)
  - Set: `export FSLDIR=/path/to/fsl`
- **AFNI**: Already installed if you preprocessed with AFNI

## Complete Workflow

### Step 0: Set Up FSL (One-Time)

The voxel-based analysis requires FSL for loading the MNI template.

**Option 1: Source FSL configuration**
```bash
source ~/.bash_profile
```

**Option 2: Add to conda environment**
```bash
mkdir -p ~/miniconda3/envs/brain_gut/etc/conda/activate.d
cat > ~/miniconda3/envs/brain_gut/etc/conda/activate.d/fsl.sh << 'EOF'
export FSLDIR=/Users/usrname/fsl
export PATH=${FSLDIR}/share/fsl/bin:${PATH}
source ${FSLDIR}/etc/fslconf/fsl.sh
EOF
chmod +x ~/miniconda3/envs/brain_gut/etc/conda/activate.d/fsl.sh

# Reactivate environment
conda deactivate && conda activate brain_gut
```

### Step 1: Check Files

Verify all required input files exist:

```bash
cd code
python synchrony_analysis/check_files_v2.py
```

**Expected output:** `Done files pre check.`

### Step 2: Prepare AFNI Data (Per Subject/Run)

**For all 15 AFNI-preprocessed subjects:**

This converts AFNI output to the format needed for synchrony analysis.

```bash
python synchrony_analysis/prepare_afni_data.py <subject> <run>
```

**Example:**
```bash
python synchrony_analysis/prepare_afni_data.py AE 1
```

**What this does:**
- Loads AFNI pb04 files (blurred, motion-corrected, MNI space)
- Creates brain mask from non-zero voxels
- Bandpass filters brain signal in gastric frequency range
- Saves filtered brain signal, mask, and preprocessed NIfTI

**Output files:**
- `derivatives/brain_gast/{subject}/{subject}{run}/func_filtered_{subject}_run{run}strict.npz`
- `derivatives/brain_gast/mask_{subject}_run{run}strict.npz`
- `derivatives/brain_gast/{subject}/{subject}{run}/{subject}_task-rest_run-0{run}_space-MNI_desc-preproc_bold_strict.nii.gz`



### Step 3: Signal Slicing (Per Subject/Run)

Temporally align gastric and brain signals:

```bash
python synchrony_analysis/signal_slicing_v2.py <subject> <run>
```

**Example:**
```bash
python synchrony_analysis/signal_slicing_v2.py AE 1
```

**What this does:**
- Loads gastric and brain signals
- Handles trigger timing and removes edge effects
- Truncates to matched length
- Saves aligned signals

**Output files:**
- `derivatives/brain_gast/{subject}/{subject}{run}/gast_data_{subject}_run{run}strict_sliced.npy`
- `derivatives/brain_gast/{subject}/{subject}{run}/func_filtered_{subject}_run{run}strict_sliced.npz`

### Step 4: Voxel-Based PLV Analysis (Per Subject/Run)

Compute phase-locking value between gastric and brain signals:

```bash
python synchrony_analysis/voxel_based_analysis_v2.py <subject> <run>
```

**Example:**
```bash
python synchrony_analysis/voxel_based_analysis_v2.py AE 1
```

**What this does:**
- Computes PLV between gastric phase and brain voxel phases
- Generates null distribution via circular permutations
- Creates statistical maps (p-values, delta, median)
- Generates visualization plots

**Output files:**
- `derivatives/brain_gast/{subject}/{subject}{run}/plvs_empirical_{subject}_run{run}.nii.gz`
- `derivatives/brain_gast/{subject}/{subject}{run}/plv_p_vals_{subject}_run{run}.nii.gz`
- `derivatives/brain_gast/{subject}/{subject}{run}/plv_delta_{subject}_run{run}.nii.gz`
- `derivatives/brain_gast/{subject}/{subject}{run}/plv_permut_median_{subject}_run{run}.nii.gz`
- `plots/brain_gast/{subject}/{subject}{run}/*.png`



### Step 5: Gastric-Motion Synchrony

Analyze relationship between gastric signals and head motion:

```bash
python synchrony_analysis/egg_confounds_synchrony_v2.py
```

Processes all subjects in metadata and creates:
- `dataframes/correl_egg_w_motion.csv`
- `dataframes/plvs_egg_w_motion.csv`
- `dataframes/motion_summary.csv`

### Step 6: Second-Level Analysis (After All Subjects)

Group-level statistical analysis using FSL randomise:

```bash
python synchrony_analysis/voxel_based_second_level_v2.py
```

**Requirements:**
- All individual subjects processed
- FSL installed and FSLDIR set


**Output:**
- Group average maps in `derivatives/brain_gast/`
- Statistical results in `derivatives/brain_gast/fsl_randomize/`

## Batch Processing

### Process All AFNI Subjects

```bash
cd code

for SUBJECT in AE AIM AlS AmK AnF AzN BS DaH DoP EdZ ElL ErG HaM IdS LA; do
    for RUN in 1 2 3; do
        if grep -q "^${SUBJECT},${RUN}," dataframes/egg_brain_meta_data_v2.csv; then
            echo "Processing ${SUBJECT} run ${RUN}..."
            python synchrony_analysis/prepare_afni_data.py $SUBJECT $RUN || continue
            python synchrony_analysis/signal_slicing_v2.py $SUBJECT $RUN || continue
            python synchrony_analysis/voxel_based_analysis_v2.py $SUBJECT $RUN
            echo "✓ Completed ${SUBJECT} run ${RUN}"
        fi
    done
done

echo "All 15 subjects processed!"
```

### Using the Batch Script

```bash
cd code
bash synchrony_analysis/run_synchrony_v2.sh
```

This interactive script walks you through all steps.

## AFNI Data Workflow

### AFNI File Structure

```
BIDS_data/soroka/sub-AE/anat_func/PreprocessedData/sub-AE/output_sub-AE/
├── pb04.sub-AE.r01.blur+tlrc.HEAD/.BRIK  (blurred - RECOMMENDED)
├── pb03.sub-AE.r01.volreg+tlrc.HEAD      (motion corrected)
└── errts.sub-AE.tproject+tlrc.HEAD       (residuals after regression)
```

### AFNI File to be used?

**Recommended: pb04 (blurred)**
- Spatially smoothed, motion corrected, MNI space
- Before nuisance regression (preserves signal)
- Best for resting-state connectivity
- Used by default in `prepare_afni_data.py`

**Alternative: errts (residuals)**
- After regression of motion/physiological confounds
- May lose some real signal
- Use with: `python synchrony_analysis/prepare_afni_data.py <subject> <run> --use-errts`

### Processing AFNI Subjects

1. **Check which subjects have AFNI data:**
   ```bash
   ls BIDS_data/soroka/
   # Output: All 15 subjects (sub-AE, sub-AIM, sub-AlS, sub-AmK, sub-AnF, sub-AzN, sub-BS, sub-DaH, sub-DoP, sub-EdZ, sub-ElL, sub-ErG, sub-HaM, sub-IdS, sub-LA)
   ```

2. **Process each subject:**
   ```bash
   cd code

   # Prepare AFNI data
   python synchrony_analysis/prepare_afni_data.py AE 1

   # Run synchrony analysis
   python synchrony_analysis/signal_slicing_v2.py AE 1
   python synchrony_analysis/voxel_based_analysis_v2.py AE 1
   ```

3. **Repeat for all runs** of all 15 AFNI subjects

## Troubleshooting

### Error: "KeyError: 'FSL_DIR'"

**Solution:** FSL environment not set. Run:
```bash
export FSLDIR=/Users/ismaila/fsl
export PATH=${FSLDIR}/share/fsl/bin:${PATH}
source ${FSLDIR}/etc/fslconf/fsl.sh
```

Or add to your conda environment activation (see Step 0).

### Error: "shape mismatch: value array of shape (X,) could not be broadcast to indexing result of shape (X,Y)"

**Solution:** This was a bug in the original script (using 4D instead of 3D shape). It's fixed in the current version. Make sure you're using the updated `voxel_based_analysis_v2.py`.

### Error: "AFNI file not found"

**Solution:** Check which AFNI files exist:
```bash
ls BIDS_data/soroka/sub-AE/anat_func/PreprocessedData/sub-AE/output_sub-AE/pb*.HEAD
```

The script tries pb04, then falls back to pb03.

### Error: "Gastric signal not found"

**Solution:** Verify gastric preprocessing completed:
```bash
ls derivatives/brain_gast/AE/AE1/*.npy
```

### Warning: "Brain and gastric signals don't match in length"

**Solution:** This is normal and handled automatically by `signal_slicing_v2.py`. The script truncates to the minimum length.

### Memory Errors

**Solution:** Process one subject at a time instead of batch processing.

## Output Structure

After processing subject AE run 1, you'll have:

```
derivatives/brain_gast/
├── AE/AE1/
│   ├── gast_data_AE_run1strict.npy                    (input: gastric)
│   ├── max_freqAE_run1strict.npy                      (input: frequency)
│   ├── func_filtered_AE_run1strict.npz                (from prepare_afni)
│   ├── gast_data_AE_run1strict_sliced.npy             (from signal_slicing)
│   ├── func_filtered_AE_run1strict_sliced.npz         (from signal_slicing)
│   ├── AE_task-rest_run-01_space-MNI_desc-preproc_bold_strict.nii.gz
│   ├── plvs_empirical_AE_run1strict.nii.gz            (from voxel_based)
│   ├── plv_p_vals_AE_run1strict.nii.gz
│   ├── plv_delta_AE_run1strict.nii.gz
│   └── plv_permut_median_AE_run1strict.nii.gz
└── mask_AE_run1strict.npz                              (from prepare_afni)

plots/brain_gast/AE/AE1/
├── egg_BOLD_sync_example.png
├── empirical_plv_map.png
└── thres95_*.png (various statistical maps)
```

## Visualization

### View Results in FSLeyes or AFNI

```bash
# FSLeyes
fsleyes derivatives/brain_gast/AE/AE1/plvs_empirical_AE_run1strict.nii.gz

# AFNI
afni derivatives/brain_gast/AE/AE1/plvs_empirical_AE_run1strict.nii.gz

# Or view PNG plots
open plots/brain_gast/AE/AE1/thres95_plvs_empirical_map.png
```

## Documentation Files

- **START_HERE.txt** - Quick start guide (read this first!)
- **README_v2.md** (this file) - Complete user guide
- **QUICKSTART_AFNI.md** - Step-by-step AFNI workflow
- **DEPENDENCIES_v2.txt** - Requirements checklist and dataset info

## Key Changes in V2

✓ Uses egg_brain_meta_data_v2.csv (15 subjects with complete data)
✓ Works with .1D motion files instead of fMRIPrep TSV
✓ Includes prepare_afni_data.py for AFNI preprocessing
✓ All paths use main_project_path from config
✓ Handles both AFNI and fMRIPrep data
✓ Resolved AIM/AlM naming inconsistency
✓ Fixed shape mismatch bug in voxel_based_analysis

## Support

For detailed AFNI workflow, see [QUICKSTART_AFNI.md](QUICKSTART_AFNI.md)
For requirements checklist, see [DEPENDENCIES_v2.txt](DEPENDENCIES_v2.txt)
For quick commands, see [START_HERE.txt](START_HERE.txt)

## Next Steps

1. Process all 15 AFNI subjects (complete dataset)
2. Run group-level analysis with all 28 runs
3. Examine results in `derivatives/brain_gast/`
4. Check plots in `plots/brain_gast/`

