'''
This script is to test 3D thermal bat tracking system

How to Run:
(tf_gpu02) python3.9 test_depth_analysis.py --left ../results/Far_Left\ 2023-07-06_18_59_59_820/Far_Left\ 2023-07-06_18_59_59_820_bat.csv --middle ../results/Middle\ 2023-07-06_18_59_59_952/Middle\ 2023-07-06_18_59_59_952_bat.csv --right ../results/Far_Right\ 2023-07-06_19_00_00_259/Far_Right\ 2023-07-06_19_00_00_259_bat.csv

Input: Left and Right csv files to analyze
Output: 3D bat/bird/insect tracking graph after false positive removal
Elapsed Time in seconds: 1486.657817363739
'''
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from smartbiofinder.model.depth_analysis import *
from smartbiofinder.utility.utils import set_directory, read_csv_file
import pandas as pd
import argparse
import time

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", help="path to the left video file")
    ap.add_argument("-m", "--middle", help="path to the middle video file")
    ap.add_argument("-r", "--right", help="path to the right video file")
    args = vars(ap.parse_args())
    left_filepath = args["left"]
    middle_filepath = args["middle"]
    right_filepath = args["right"]

    startTime = time.time()             # Record the elapsed time for the process

    # Read dataframes (2D Bat Tracking Output from each video)
    df_left, df_middle, df_right, left_file, middle_file, right_file = read_csv_file(left_filepath, middle_filepath, right_filepath)
    print("Filenames: ", left_file, middle_file, right_file)               # e.g.) Far_Left 2023-07-06_18_59_59_820_bat Middle 2023-07-06_18_59_59_952_bat Far_Right 2023-07-06_19_00_00_259_bat

    assert left_file[-3:] == middle_file[-3:] == right_file[-3:]        # ends with 'bat'
    assert left_file.split(" ")[-1].split("_")[0] == middle_file.split(" ")[-1].split("_")[0] == right_file.split(" ")[-1].split("_")[0]        # same dates
    date = left_file.split(" ")[-1].split("_")[0]

    # Estimate the frame drift
    frame_diff_lm = calculate_frame_drift(30.0, left_file, middle_file)
    print("frame difference (middle-left): ", frame_diff_lm)

    frame_diff_mr = calculate_frame_drift(30.0, middle_file, right_file)
    print("frame difference (right-middle): ", frame_diff_mr)

    # Path for saving results
    new_filename = date + '_bat'
    save_dir = '../depth_analysis/'+new_filename
    set_directory('../depth_analysis')
    set_directory(save_dir)
    os.chdir(save_dir)

    # Dataframe to save the results
    df_bats = create_df_analysis()

    # Far_Left & Middle => Right is standard
    # Middle & Far_Right => Left is standard
    df_bats = match_frames_lm(df_left, df_middle, frame_diff_lm, df_bats)
    df_bats = match_frames_mr(df_middle, df_right, frame_diff_mr, df_bats)
    
    df_bats_sorted = df_bats.sort_values(by=['Middle Image'])
    df_bats_sorted.to_excel(f'../Full_bat_{new_filename}.xlsx')
    endTime = time.time() 
    executionTime = endTime - startTime
    print('Execution time in seconds: ' + str(executionTime))


if __name__ == "__main__":
    main()