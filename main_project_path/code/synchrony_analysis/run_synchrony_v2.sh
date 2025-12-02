#!/bin/bash

# Synchrony Analysis V2 - Batch Processing Script
# This script runs the synchrony analysis pipeline for all subjects

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "========================================="
echo "Synchrony Analysis V2 Pipeline"
echo "========================================="

# Step 1: Check files
echo ""
echo "[Step 1/5] Checking required files..."
python synchrony_analysis/check_files_v2.py
if [ $? -eq 0 ]; then
    echo "✓ All required files found!"
else
    echo "✗ Some files are missing. Please check the errors above."
    exit 1
fi

# Step 2: Compute EGG-Motion synchrony (optional)
echo ""
echo "[Step 2/5] Computing EGG-Motion synchrony..."
read -p "Do you want to compute gastric-motion synchrony? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python synchrony_analysis/egg_confounds_synchrony_v2.py
    echo "✓ EGG-Motion synchrony completed!"
else
    echo "Skipping EGG-Motion synchrony analysis."
fi

# Step 3: Check if brain files exist OR offer to prepare AFNI data
echo ""
echo "[Step 3/5] Checking for brain signal files..."
BRAIN_FILES_EXIST=false
AFNI_AVAILABLE=false

# Check a few examples
if [ -f "../derivatives/brain_gast/AE/AE1/func_filtered_AE_run1strict.npz" ]; then
    echo "✓ Brain signal files found!"
    BRAIN_FILES_EXIST=true
elif [ -d "../BIDS_data/soroka/sub-AE" ]; then
    echo "⚠ Brain signal files not found, but AFNI preprocessing detected"
    AFNI_AVAILABLE=true
fi

if [ "$BRAIN_FILES_EXIST" = false ] && [ "$AFNI_AVAILABLE" = true ]; then
    echo ""
    echo "AFNI preprocessing available for: ALL 15 subjects"
    echo ""
    read -p "Do you want to prepare AFNI data now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Preparing AFNI data for all 15 subjects..."
        for SUBJECT in AE AIM AlS AmK AnF AzN BS DaH DoP EdZ ElL ErG HaM IdS LA; do
            while IFS=, read -r sub run rest; do
                if [ "$sub" == "subject" ]; then continue; fi
                if [ "$sub" == "$SUBJECT" ]; then
                    echo "  Processing $SUBJECT run $run..."
                    python synchrony_analysis/prepare_afni_data.py "$SUBJECT" "$run" || continue
                fi
            done < dataframes/egg_brain_meta_data_v2.csv
        done
        echo "✓ AFNI data preparation completed for all 15 subjects!"
        BRAIN_FILES_EXIST=true
    else
        echo "Skipping AFNI data preparation."
        echo "You can run it manually with:"
        echo "  python synchrony_analysis/prepare_afni_data.py <subject> <run>"
        exit 0
    fi
elif [ "$BRAIN_FILES_EXIST" = false ]; then
    echo ""
    echo "⚠ Brain preprocessing files are missing and no AFNI data found."
    echo "You can only run EGG-Motion synchrony analysis (Step 2)."
    echo "To run voxel-based analysis, you need to:"
    echo "  1. Run brain preprocessing pipeline (AFNI or fMRIPrep)"
    echo "  2. Generate brain signal files with prepare_afni_data.py"
    exit 0
fi

# Step 4: Signal slicing and voxel-based analysis (per subject)
echo ""
echo "[Step 4/5] Processing subjects..."
read -p "Do you want to run signal slicing and voxel-based analysis? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping per-subject analysis."
    exit 0
fi

# Read subjects from CSV and process
PROCESSED=0
FAILED=0

while IFS=, read -r subject run rest; do
    # Skip header
    if [ "$subject" == "subject" ]; then
        continue
    fi

    echo ""
    echo "Processing: $subject run $run"

    # Signal slicing
    if python synchrony_analysis/signal_slicing_v2.py "$subject" "$run"; then
        echo "  ✓ Signal slicing completed"
    else
        echo "  ✗ Signal slicing failed"
        ((FAILED++))
        continue
    fi

    # Voxel-based analysis
    if python synchrony_analysis/voxel_based_analysis_v2.py "$subject" "$run"; then
        echo "  ✓ Voxel-based analysis completed"
        ((PROCESSED++))
    else
        echo "  ✗ Voxel-based analysis failed"
        ((FAILED++))
    fi

done < dataframes/egg_brain_meta_data_v2.csv

echo ""
echo "Per-subject processing complete!"
echo "  Processed: $PROCESSED subject-runs"
echo "  Failed: $FAILED subject-runs"

# Step 5: Second-level analysis
echo ""
echo "[Step 5/5] Group-level analysis..."

if [ $PROCESSED -eq 0 ]; then
    echo "No subjects were successfully processed. Skipping group analysis."
    exit 0
fi

read -p "Do you want to run second-level (group) analysis? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python synchrony_analysis/voxel_based_second_level_v2.py
    echo "✓ Second-level analysis completed!"
else
    echo "Skipping second-level analysis."
fi

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="