'''
This script is to test 3D thermal bat tracking system

How to Run:
(tf_gpu02) python3.9 test_depth_analysis.py --left ../results/Middle\ 2023-07-06_21_50_10/Middle\ 2023-07-06_21_50_10_bat.csv --right ../results/Right\ 2023-07-06_21_50_00/Right\ 2023-07-06_21_50_00_bat.csv

Input: Left and Right csv files to analyze
Output: 3D bat/bird/insect tracking graph after false positive removal
'''
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from smartbiofinder.model.depth_analysis import *
from smartbiofinder.utility.utils import set_directory
import numpy as np
import pandas as pd
import argparse
import cv2
import time


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", help="path to the left video file")
    ap.add_argument("-r", "--right", help="path to the right video file")
    args = vars(ap.parse_args())
    left_filepath = args["left"]
    right_filepath = args["right"]
    frame_diff_range =  30 * 10         # 10 seconds    # For 5mins) 30 * 300     # Assuming 30fps for 5mins

    startTime = time.time()             # Record the elapsed time for the process

    # Read dataframes (2D Bat Tracking Output from each video)
    df_left = pd.read_csv(left_filepath)
    df_right = pd.read_csv(right_filepath)

    frame_col_left = df_left.loc[:, ['Current Frame']]      # originally ['Frame Count']
    frame_col_right = df_right.loc[:, ['Current Frame']]
    max_num_right = frame_col_right.values[len(frame_col_right)-1][0]
    
    left_file, separator, extension = left_filepath.split("/")[-1].partition('.')
    right_file, separator, extension = right_filepath.split("/")[-1].partition('.')
    print("left, right: ", left_file, right_file)

    # Read calibration matrix
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = read_calibration_matrix()

    # Get frame width/height
    width = 800
    height = 600

    # Visualization Specification
    radius = 25         # Radius of circle
    color = (255, 255, 255)     # White color in BGRn
    thickness = 2       # -1          # Fill in circle

    # Camera Intrinsic Parameters
    f = 19      # focal length = 19mm
    pixel_size = 0.017      # pixel size = 0.017mm

    # Path for saving results
    new_filename = left_file + '_' + right_file
    save_dir = '../depth_analysis/'+new_filename
    set_directory('../depth_analysis')
    set_directory(save_dir)
    os.chdir(save_dir)

    # Find corresponding frame matches
    for idx_left in range(len(frame_col_left)):
        num = frame_col_left.values[idx_left][0]     # frame number in the left camera
        frame_match_candidate = {}
        for num_right in range(max(0, num-frame_diff_range), min(max_num_right, num+frame_diff_range)+1):
            right_row = df_right.loc[(df_right['Current Frame'] == num_right)]
            if right_row.empty:
                continue
            left_row = df_left.loc[(df_left['Current Frame'] == num)]
            right_object = right_row.iloc[0]['Object Name']
            left_object = left_row.iloc[0]['Object Name']
            right_id = right_row.iloc[0]['Object ID']
            left_id = left_row.iloc[0]['Object ID']
            left_pre = left_row.iloc[0]['Probability']
            right_pre = right_row.iloc[0]['Probability']
            left_area = left_row.iloc[0]['Area']
            right_area = right_row.iloc[0]['Area']
            left_cnt = read_cnt_from_csv(left_row.iloc[0]['Detected Contour Array(px)'])
            right_cnt = read_cnt_from_csv(right_row.iloc[0]['Detected Contour Array(px)'])

            # left_bounding_box = [left_row.iloc[0]['X'],left_row.iloc[0]['Y'],left_row.iloc[0]['Width'],left_row.iloc[0]['Height']]
            # right_bounding_box = [right_row.iloc[0]['X'],right_row.iloc[0]['Y'],right_row.iloc[0]['Width'],right_row.iloc[0]['Height']]

            # Extract cX and cY values
            # left_center = np.zeros((left_row.iloc[0]['Center Y'], left_row.iloc[0]['Center X'], 3))
            # right_center = np.zeros((right_row.iloc[0]['Center Y'], right_row.iloc[0]['Center X'], 3))

            # Create contour masks to perform rectification for accurate pixel disparity 
            left_mask = np.zeros([height, width, 3],dtype='uint8')
            right_mask = np.zeros([height, width, 3],dtype='uint8') 
            cv2.drawContours(left_mask, [left_cnt], -1, (255,255,255), thickness=cv2.FILLED)
            cv2.drawContours(right_mask, [right_cnt], -1, (255,255,255), thickness=cv2.FILLED)
            
            left_frame = cv2.imread('../../results/' + left_row.iloc[0]['File Path'])
            right_frame = cv2.imread('../../results/' + right_row.iloc[0]['File Path'])

            # Rectify frames with contours - shape transformation: (600, 800, 3) -> (480, 640, 3)
            rectified_left, rectified_right = rectify_frames(left_frame, right_frame, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)       # frame shape: (600, 800, 3)
            rectified_left_mask, rectified_right_mask = rectify_frames(left_mask, right_mask, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
            
            rectified_with_mask_left = cv2.add(rectified_left, rectified_left_mask)
            rectified_with_mask_right = cv2.add(rectified_right, rectified_right_mask)

            # Frame matching
            # 1) similar y-axis value
            M_left = cv2.moments(cv2.cvtColor(rectified_left_mask, cv2.COLOR_BGR2GRAY))
            if M_left["m00"]==0:
                continue
            cX_left = int(M_left["m10"]/M_left["m00"])        				
            cY_left = int(M_left["m01"]/M_left["m00"])

            M_right = cv2.moments(cv2.cvtColor(rectified_right_mask, cv2.COLOR_BGR2GRAY))
            if M_right["m00"]==0:
                continue
            cX_right = int(M_right["m10"]/M_right["m00"])        				
            cY_right = int(M_right["m01"]/M_right["m00"])

            if cY_right > cY_left + 80 or cY_right < cY_left+40:        # Epipolar not horizontal.. -> not frame matched, and hence pass
                continue
            
            # 2) contour template matching
            if cY_left+15 > 480 or cY_right+15 > 480 or cY_left-15 < 0 or cY_right-15 < 0 or cX_left+15 > 640 or cX_right+15 > 640 or cX_left-15 < 0 or cX_right-15 < 0:
                continue
            
            frame_match_left = rectified_left[cY_left - 15 : cY_left + 15, cX_left - 15 : cX_left + 15]
            frame_match_right = rectified_right[cY_right - 15 : cY_right + 15, cX_right - 15 : cX_right + 15]
            
            result = cv2.matchTemplate(frame_match_right, frame_match_left, cv2.TM_CCOEFF_NORMED)
            if result[0][0] > 0.8:
                left_center = (cX_left, cY_left)
                right_center = (cX_right, cY_right)
                frame_match_candidate[(num_right, left_center, right_center, left_area, right_area)] = (result[0][0], rectified_with_mask_left, rectified_with_mask_right)
                print(num, num_right, result[0][0])
        
        # Depth Analysis on best frame matches
        if frame_match_candidate == {}:
            continue
        
        best_num_right, best_left_center, best_right_center, best_left_area, best_right_area = max(frame_match_candidate, key=lambda x: frame_match_candidate[x][0])
        _, best_rectified_with_mask_left, best_rectified_with_mask_right = frame_match_candidate[(best_num_right, best_left_center, best_right_center, best_left_area, best_right_area)]
        
        # Estimate depth
        depth, norm_depth = find_depth(best_right_center, best_left_center)

        # Estimate actual object size
        inch_depth = depth * 12                 # ft -> in
        inch_norm_depth = norm_depth * 12       # ft -> in

        estimate_real_area = int(((inch_depth**2) * ((best_left_area + best_right_area) / 2) * (pixel_size**2)) / (f**2))
        norm_estimate_real_area = int(((inch_norm_depth**2) * ((best_left_area + best_right_area) / 2) * (pixel_size**2)) / (f**2))

        # Save rectified matched frames
        stacked = np.hstack([np.array(best_rectified_with_mask_left), np.array(best_rectified_with_mask_right)])
        cv2.putText(stacked, f'{depth}ft : {estimate_real_area}in^2 - linear {norm_depth}ft : {norm_estimate_real_area}in^2',(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite('rectified_frame_'+str(num)+'_'+str(best_num_right)+'.jpg', stacked)

        # Dump into csv file

    
    endTime = time.time() 
    executionTime = endTime - startTime
    print('Execution time in seconds: ' + str(executionTime))


'''
    object_lst = ['bat','bird','insect']
    headers = {'Object Type':[], 'Left Center (px)':[], 'Right Center (px)':[], 'Depth Prediction (cm)':[],'Linearized Depth Prediction':[], 'Left Multi Predictions':[], 'Right Multi Predictions':[],  'Left Detected Countour Array (px)':[], 'Right Detected Countour Array (px)':[], 'Left Rect Bounding Box [x,y,w,h] (px)':[], 'Right Rect Bounding Box [x,y,w,h] (px)':[], 'Left Contour Area (px)':[], 'Right Contour Area (px)':[], 'Estimated Contour Area (cm^2)':[], 'Estimated Contour Area with Norm Depth (cm^2)':[], 'Hour of day':[],'Left Image':[],'Right Image':[]}
    df_bats = pd.DataFrame(headers)

    # Visualization Specification
    radius = 20         # Radius of circle
    color = (255, 255, 255)     # White color in BGRn
    thickness = -1          # Fill in circle

    # Let's assume that frames are synchronized
    for k in range(len(frame_col_left)):
        num = frame_col_left.values[k][0]     # frame number in the left camera
        num_right = num-16
        right_row = df_right.loc[(df_right['Current Frame'] == num_right)]
        if right_row.empty:
            continue
        
        left_row = df_left.loc[(df_left['Current Frame'] == num)]
        right_object = right_row.iloc[0]['Object Name']
        left_object = left_row.iloc[0]['Object Name']
        right_id = right_row.iloc[0]['Object ID']
        left_id = left_row.iloc[0]['Object ID']
        left_pre = left_row.iloc[0]['Probability']
        right_pre = right_row.iloc[0]['Probability']
        cnt_left = left_row.iloc[0]['Detected Contour Array(px)']
        cnt_right = right_row.iloc[0]['Detected Contour Array(px)']
        # l_vid_file = left_row.iloc[0]['Filename']
        # r_vid_file = right_row.iloc[0]['Filename']
        l_vid_file = "Middle 2023-07-06_21_50_10.mp4"
        r_vid_file = "Right 2023-07-06_21_50_00.mp4"

        left_bounding_box = [left_row.iloc[0]['X'],left_row.iloc[0]['Y'],left_row.iloc[0]['Width'],left_row.iloc[0]['Height']]
        right_bounding_box = [right_row.iloc[0]['X'],right_row.iloc[0]['Y'],right_row.iloc[0]['Width'],right_row.iloc[0]['Height']]
        
        cap_left_object = left_object.capitalize()
        cap_right_object = right_object.capitalize()

        left_path = f'../data/2023-07-06 bat_identified/out_low_cnt_Middle 2023-07-06_21_50_10/{left_object}_detected/{left_file}_{left_object}_object_id_{left_id}/{left_object}_{left_id}_{left_file}_ frame_num_{num}.jpg'
        right_path = f'../data/2023-07-06 bat_identified/out_low_cnt_Right 2023-07-06_21_50_00/{right_object}_detected/{right_file}_{right_object}_object_id_{right_id}/{right_object}_{right_id}_{right_file}_ frame_num_{num_right}.jpg'
        print('path: ', left_path, right_path)

        # ref_left = cv2.imread(left_path)
        # ref_right = cv2.imread(right_path)

        # cv2.imshow('ref_left', ref_left)
        # cv2.imshow('ref_right', ref_right)

        # Extract cX and cY values
        right_center = (right_row.iloc[0]['Center X'], right_row.iloc[0]['Center Y'])
        left_center = (left_row.iloc[0]['Center X'], left_row.iloc[0]['Center Y'])

        # Create masks to perform rectification for accurate pixel disparity 
        right_mask = np.zeros([480, 640],dtype='uint8')
        left_mask = np.zeros([480, 640],dtype='uint8') 
        right_frame = cv2.circle(right_mask, right_center, radius, color, thickness)
        left_frame = cv2.circle(left_mask, left_center, radius, color, thickness)
        # cv2.imshow('not rectified right', right_frame)
        # cv2.imshow('not rectified left', left_frame)

        left_frame = left_frame[20:480, 0:640]
        right_frame = right_frame[20:480, 0:640]
        rectified_left_frame = cv2.remap(left_frame, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rectified_right_frame = cv2.remap(right_frame, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
        left_roi = rectified_left_frame.copy()
        right_roi = rectified_right_frame.copy()

        depth, norm_depth = bf.find_depth(right_center, left_center, right_roi, left_roi, depth_lst, right_lst, left_lst)
        if depth == np.inf:         # Focus on Rotor Swept area which is height > 40m
            print("Ignore Depth: ", depth)
            continue
        

        depth, norm_depth = int(depth), int(norm_depth)
        
        cnt_left, left_area = bf.excel_cnt_to_area(cnt_left)
        cnt_right, right_area = bf.excel_cnt_to_area(cnt_right)

        if abs(left_area - right_area) > 500:
            print("Contour not matching..")
            continue

        estimate_time = bf.frame_to_time(num)
        
        cv2.putText(left_roi, f'{left_object} -{depth} - {norm_depth}',(20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(right_roi, f'{right_object} - {depth} - {norm_depth}',(20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print("num: ", num)
        
        # cv2.imshow('rectified right', right_roi)
        # cv2.imshow('rectified left', left_roi)
        print(f'INFO: depth {depth}, \n norm_depth {norm_depth}, \n left_pre {left_pre} ,\n right_pre {right_pre},\n estimate_time {estimate_time}')
        print(f'Left area {left_area}, Right area {right_area}')

        f = 19      # focal length = 19mm
        pixel_size = 0.017      # pixel size = 0.017mm

        estimate_real_area = ((depth**2) * ((left_area + right_area) / 2) * (pixel_size**2)) / (f**2)
        norm_estimate_real_area = ((norm_depth**2) * ((left_area + right_area) / 2) * (pixel_size**2)) / (f**2)

        df_bats.loc[len(df_bats.index)] = ['bat', left_center, right_center, depth, norm_depth, left_pre , right_pre, cnt_left, cnt_right, left_bounding_box, right_bounding_box, left_area, right_area, estimate_real_area, norm_estimate_real_area, estimate_time, left_path, right_path]
        

    # Write a dumped file 
    new_filename = left_file + '_' + right_file
    df_bats.to_excel(f'../output/Full_{object_lst[0]}_training_{new_filename}.xlsx')

    # Check stereo rectification result - see if having epipolar line
'''



if __name__ == "__main__":
    main()