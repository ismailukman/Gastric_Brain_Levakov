import pandas as pd
import os
import sys
import pathlib

# 1) Add the parent directory of the current script to Python's module search path
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))

# 2) Change working directory to the project's root (one level above this script)
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))

# 3) Import configuration variables: clean_level (data cleaning setting) and main_project_path (base path to project data)
from config import clean_level, main_project_path

# 4) Load subject and run metadata from CSV file (contains info about subjects, runs, exclusions, etc.)
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data_v2.csv')

# 5) Apply filtering if strict cardiac cleaning mode is enabled:
#     - Keep only entries where 'ppu_exclude' is False (not excluded)
#     - Keep only entries where 'ppu_found' is True (PPU file found)
if clean_level == 'strict_gs_cardiac':
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_exclude'] == False, :]
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]

# 6) Iterate through each subject-run entry in the metadata
for run_indx in record_meta_pd.index.values:
    data_paths = []
    subject_name = record_meta_pd.loc[run_indx,'subject']
    run = str(record_meta_pd.loc[run_indx,'run'])

    # 7) Construct file paths for motion parameters (.1D) and gastric signal (.npy) files
    # Motion file path: BIDS_data/sub_motion_files/sub-{subject}_dfile.r0{run}.1D
    motion_file_path = f'{main_project_path}/BIDS_data/sub_motion_files/sub-{subject_name}_dfile.r0{run}.1D'
    data_paths.append(motion_file_path)

    # Gastric signal file path: derivatives/brain_gast/{subject}/{subject}{run}/gast_data_{subject}_run{run}{clean_level}.npy
    gastric_file_path = f'{main_project_path}/derivatives/brain_gast/{subject_name}/{subject_name}{run}/gast_data_{subject_name}_run{run}{clean_level}.npy'
    data_paths.append(gastric_file_path)

    # Max frequency file path (also in derivatives)
    freq_file_path = f'{main_project_path}/derivatives/brain_gast/{subject_name}/{subject_name}{run}/max_freq{subject_name}_run{run}{clean_level}.npy'
    data_paths.append(freq_file_path)

    # 8) Verify that all expected files exist; raise an error if any is missing
    for data_path in data_paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'file not found: {data_path}')

# 9) Print confirmation message when all files are successfully verified
print('Done files pre check.')