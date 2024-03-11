from smartbiofinder.utility.utils_cv import *
import cv2
import math
import numpy as np
from datetime import datetime, timedelta

def read_calibration_matrix():
    cv_file = cv2.FileStorage() #file must be in same folder as the videos
    cv_file.open('../../calibration/stereoMap_16.xml', cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y


def rectify_frames(left_frame, right_frame, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y):

    # Undistort and rectify images
    undistorted_left = cv2.remap(left_frame, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistorted_right = cv2.remap(right_frame, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return undistorted_left, undistorted_right


def calculate_frame_drift(fps, left_file, right_file):
    # Estimate the frame drift
    left_start = datetime.strptime(left_file.split(" ")[-1][:-4], '%Y-%m-%d_%H_%M_%S_%f')
    right_start = datetime.strptime(right_file.split(" ")[-1][:-4], '%Y-%m-%d_%H_%M_%S_%f')
    delta = right_start - left_start
    frame_diff = fps * delta.total_seconds()

    return frame_diff


def find_depth(right_point, left_point):
    # M = np.array([[8446.55113357],
    #               [ -57.46447252]])
    M = 8446.55113357 #ft
    baseline = 99      # Distance between the cameras [cm] -> need to verify: (changed 95.25 to 99cm after actual measurement)
    f = 19             # Camera lense's focal length [mm]
    alpha = 19.4       # Camera field of view in the horizontal plane [degrees]
    mean = 173.1121542
    std = 34.94402137
    normalized = 0.876034097 #This was found at 255 feet in the truck example

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    width_right = 640
    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)     # 1872.0771061225453

    x_rightd = right_point[0]
    x_leftd = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_leftd-x_rightd      #Displacement between left and right frames [pixels]
    if disparity == 0:
        return 0
    
    standardized_disparity = (disparity-mean)/std
    normalized_disparity = standardized_disparity + normalized

    zDepth = ((baseline * f_pixel) / disparity)         # return 'cm' unit zDepth

    return abs(zDepth)
    
    # CALCULATE DEPTH Z:
    # zDepth = ((baseline * f_pixel) / disparity) * 0.032808         # cm -> feet (by multiplying 0.032808)   # '*3' has been added -> need to be fixed in the future
    # zDepthwithM = (M / disparity) * 0.032808                            # To use M, the camera calibration and stereo rectification should be perfect first.
    # linearized_depth = (0.5877 + normalized_disparity)/0.0213       # feet

    # print("zDepth, zDepthwithM, linearized depth")

    # if linearized_depth > 100 or zDepth > 100:
    #     return abs(zDepthwithM)


def estimate_size(left_center, right_center, left_area, right_area):
    # Camera Intrinsic Parameters
    f = 19      # focal length = 19mm
    pixel_size = 0.017      # pixel size = 0.017mm

    # Estimate depth and size
    depth = find_depth(right_center, left_center)     # cm
    estimate_real_area = int((((depth**2) * ((left_area + right_area) / 2) * (pixel_size**2)) / (f**2)) * 0.155)        # in^2 (1cm^2 = 0.155in^2)
    
    return depth, estimate_real_area



def match_frames_mr(df_middle, df_right, frame_diff, df_bats):
    frame_col_middle = df_middle.loc[:, ['Current Frame']]      # originally ['Frame Count']
    adjust_value = 56       # Post processing of the rectification results - Need to cut off left portion of the left frames

    # Read calibration matrix
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = read_calibration_matrix()

    # Find corresponding frame matches
    for idx_middle in range(len(frame_col_middle)):
        num = frame_col_middle.values[idx_middle][0]     # frame number in the left camera
        middle_row = df_middle.loc[(df_middle['Current Frame'] == num)]
        saved = False
        
        # Right = Middle - frame_diff
        for num_right in [num-math.floor(frame_diff), num-math.ceil(frame_diff)]:       # Have 2 options
            if saved == True:
                continue
            
            right_row = df_right.loc[(df_right['Current Frame'] == num_right) & (df_right['Center Y'] > (middle_row.iloc[0]['Center Y']+40)) & (df_right['Center Y'] < (middle_row.iloc[0]['Center Y']+80))]     # condition: cY_left+40 < cY_right < cY_left+80
            if right_row.empty:
                continue
            
            middle_id = middle_row.iloc[0]['Object ID']
            middle_pre = middle_row.iloc[0]['Probability']
            right_pre = right_row.iloc[0]['Probability']
            middle_area = middle_row.iloc[0]['Area']
            right_area = middle_row.iloc[0]['Area']
            middle_cnt = read_cnt_from_csv(middle_row.iloc[0]['Detected Contour Array(px)'])
            right_cnt = read_cnt_from_csv(right_row.iloc[0]['Detected Contour Array(px)'])
            middle_path = middle_row.iloc[0]['File Path']
            right_path = right_row.iloc[0]['File Path']
            estimate_time = middle_row.iloc[0]['Frame Timestamp']

            if len(middle_cnt) == 0 or len(right_cnt) == 0:
                print("Having not appropriate contour arrays..")
                continue
            
            if abs(middle_area - right_area) > 1000:
                print("Probably not matching area..")
                continue
            
            # Create contour masks to perform rectification for accurate pixel disparity
            width, height = 800, 600
            middle_mask = np.zeros([height, width, 3],dtype='uint8')
            right_mask = np.zeros([height, width, 3],dtype='uint8') 
            cv2.drawContours(middle_mask, [middle_cnt], -1, (255,255,255), thickness=cv2.FILLED)
            cv2.drawContours(right_mask, [right_cnt], -1, (255,255,255), thickness=cv2.FILLED)
            
            middle_frame = cv2.imread('../../results/' + middle_row.iloc[0]['File Path'])
            right_frame = cv2.imread('../../results/' + right_row.iloc[0]['File Path'])

            # Rectify frames with contours - shape transformation: (600, 800, 3) -> (480, 640, 3)
            rectified_middle, rectified_right = rectify_frames(middle_frame, right_frame, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)       # frame shape: (600, 800, 3)
            rectified_middle_mask, rectified_right_mask = rectify_frames(middle_mask, right_mask, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
            
            rectified_with_mask_middle = cv2.add(rectified_middle, rectified_middle_mask)
            rectified_with_mask_right = cv2.add(rectified_right, rectified_right_mask)

            M_middle = cv2.moments(cv2.cvtColor(rectified_middle_mask, cv2.COLOR_BGR2GRAY))
            if M_middle["m00"]==0:
                continue
            cX_middle = int(M_middle["m10"]/M_middle["m00"])        				
            cY_middle = int(M_middle["m01"]/M_middle["m00"])

            M_right = cv2.moments(cv2.cvtColor(rectified_right_mask, cv2.COLOR_BGR2GRAY))
            if M_right["m00"]==0:
                continue
            cX_right = int(M_right["m10"]/M_right["m00"])        				
            cY_right = int(M_right["m01"]/M_right["m00"])

            middle_center = (cX_middle, cY_middle)
            right_center = (cX_right, cY_right)

            # Post processing of the rectification results - Need to cut off left portion of the left frames
            middle_center = (middle_center[0] - adjust_value, middle_center[1])
            depth, estimate_real_area = estimate_size(middle_center, right_center, middle_area, right_area)

            if depth == 0 or estimate_real_area < 40 or estimate_real_area > 1000:       # filter out with estimated size
                continue

            # Save rectified matched frames
            stacked = np.hstack([np.array(rectified_with_mask_middle), np.array(rectified_with_mask_right)])
            mask = np.zeros(stacked.shape, dtype='uint8')
            mask[:, :] = [255, 255, 255]
            mask = cv2.rectangle(mask, (0, 0), (adjust_value, stacked.shape[0]), (0, 0, 0), -1)
            stacked = cv2.bitwise_and(stacked, mask)

            depth += 190.5          # add sensor height (75inch = 190.5cm)
            cv2.putText(stacked, f'{int(depth)}cm : {estimate_real_area}in^2',(680, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imwrite('rectified_frame_m_'+str(num)+'_r_'+str(num_right)+'.jpg', stacked)
            saved = True

        # Dump into csv file
        ##### Put object id, time #####
        # headers = {'Object Type':[], 'Object ID':[], 'Rectified Left Center (px)':[], 'Rectified Middle Center (px)':[], 'Rectified Right Center (px)':[], 'Depth Prediction (cm)':[], 'Left Multi Predictions':[], 'Middle Multi Predictions':[], 'Right Multi Predictions':[],  'Left Detected Countour Array (px)':[], 'Middle Detected Countour Array (px)':[], 'Right Detected Countour Array (px)':[], 'Left Contour Area (px)':[], 'Middle Contour Area (px)':[], 'Right Contour Area (px)':[], 'Estimated Contour Area (in^2)':[], 'Time':[], 'Left Image':[], 'Middle Image':[], 'Right Image':[]}
        if saved == True:
            df_bats.loc[len(df_bats.index)] = ['bat', middle_id, '-', middle_center, right_center, depth, '-', middle_pre , right_pre, '-', middle_cnt, right_cnt, '-', middle_area, right_area, estimate_real_area, estimate_time, '-', middle_path, right_path]
    

    return df_bats


def match_frames_lm(df_left, df_middle, frame_diff, df_bats):
    frame_col_middle = df_middle.loc[:, ['Current Frame']]      # originally ['Frame Count']
    adjust_value = 240       # Post processing of the rectification results - Need to cut off left portion of the left frames

    # Read calibration matrix
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = read_calibration_matrix()

    # Find corresponding frame matches
    for idx_middle in range(len(frame_col_middle)):
        num = frame_col_middle.values[idx_middle][0]     # frame number in the left camera
        middle_row = df_middle.loc[(df_middle['Current Frame'] == num)]
        saved = False
        
        # Middle = Left - frame_diff
        # Left = Middle + frame_diff        # doesn't matter whether left started earlier or not
        for num_left in [num+math.floor(frame_diff), num+math.ceil(frame_diff)]:
            if saved == True:
                continue
            
            left_row = df_left.loc[(df_left['Current Frame'] == num_left) & (df_left['Center Y'] > (middle_row.iloc[0]['Center Y']-80)) & (df_left['Center Y'] < (middle_row.iloc[0]['Center Y']-40))]     # condition: cY_left+40 < cY_right < cY_left+80
            if left_row.empty:
                continue
            
            middle_id = middle_row.iloc[0]['Object ID']
            left_pre = left_row.iloc[0]['Probability']
            middle_pre = middle_row.iloc[0]['Probability']
            left_area = left_row.iloc[0]['Area']
            middle_area = middle_row.iloc[0]['Area']
            left_cnt = read_cnt_from_csv(left_row.iloc[0]['Detected Contour Array(px)'])
            middle_cnt = read_cnt_from_csv(middle_row.iloc[0]['Detected Contour Array(px)'])
            left_path = left_row.iloc[0]['File Path']
            middle_path = middle_row.iloc[0]['File Path']
            estimate_time = left_row.iloc[0]['Frame Timestamp']

            if len(left_cnt) == 0 or len(middle_cnt) == 0:
                print("Having not appropriate contour arrays..")
                continue
            
            if abs(left_area - middle_area) > 1000:
                print("Probably not matching area..")
                continue
            
            # Create contour masks to perform rectification for accurate pixel disparity
            width, height = 800, 600
            left_mask = np.zeros([height, width, 3],dtype='uint8')
            middle_mask = np.zeros([height, width, 3],dtype='uint8') 
            cv2.drawContours(left_mask, [left_cnt], -1, (255,255,255), thickness=cv2.FILLED)
            cv2.drawContours(middle_mask, [middle_cnt], -1, (255,255,255), thickness=cv2.FILLED)
            
            left_frame = cv2.imread('../../results/' + left_row.iloc[0]['File Path'])
            middle_frame = cv2.imread('../../results/' + middle_row.iloc[0]['File Path'])

            # Rectify frames with contours - shape transformation: (600, 800, 3) -> (480, 640, 3)
            rectified_left, rectified_middle = rectify_frames(left_frame, middle_frame, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)       # frame shape: (600, 800, 3)
            rectified_left_mask, rectified_middle_mask = rectify_frames(left_mask, middle_mask, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
            
            rectified_with_mask_left = cv2.add(rectified_left, rectified_left_mask)
            rectified_with_mask_middle = cv2.add(rectified_middle, rectified_middle_mask)

            M_left = cv2.moments(cv2.cvtColor(rectified_left_mask, cv2.COLOR_BGR2GRAY))
            if M_left["m00"]==0:
                continue
            cX_left = int(M_left["m10"]/M_left["m00"])        				
            cY_left = int(M_left["m01"]/M_left["m00"])

            M_middle = cv2.moments(cv2.cvtColor(rectified_middle_mask, cv2.COLOR_BGR2GRAY))
            if M_middle["m00"]==0:
                continue
            cX_middle = int(M_middle["m10"]/M_middle["m00"])        				
            cY_middle = int(M_middle["m01"]/M_middle["m00"])

            left_center = (cX_left, cY_left)
            middle_center = (cX_middle, cY_middle)

            # Post processing of the rectification results - Need to cut off left portion of the left frames
            left_center = (left_center[0] - adjust_value, left_center[1])
            depth, estimate_real_area = estimate_size(left_center, middle_center, left_area, middle_area)

            if depth == 0 or estimate_real_area < 40 or estimate_real_area > 1000:       # filter out with estimated size
                continue

            # Save rectified matched frames
            stacked = np.hstack([np.array(rectified_with_mask_left), np.array(rectified_with_mask_middle)])
            mask = np.zeros(stacked.shape, dtype='uint8')
            mask[:, :] = [255, 255, 255]
            mask = cv2.rectangle(mask, (0, 0), (adjust_value, stacked.shape[0]), (0, 0, 0), -1)
            stacked = cv2.bitwise_and(stacked, mask)

            depth += 190.5          # add sensor height (75inch = 190.5cm)
            cv2.putText(stacked, f'{int(depth)}cm : {estimate_real_area}in^2',(680, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imwrite('rectified_frame_l_'+str(num_left)+'_m_'+str(num)+'.jpg', stacked)
            saved = True

        # Dump into csv file
        ##### Put object id, time #####
        # headers = {'Object Type':[], 'Object ID':[], 'Rectified Left Center (px)':[], 'Rectified Middle Center (px)':[], 'Rectified Right Center (px)':[], 'Depth Prediction (cm)':[], 'Left Multi Predictions':[], 'Middle Multi Predictions':[], 'Right Multi Predictions':[],  'Left Detected Countour Array (px)':[], 'Middle Detected Countour Array (px)':[], 'Right Detected Countour Array (px)':[], 'Left Contour Area (px)':[], 'Middle Contour Area (px)':[], 'Right Contour Area (px)':[], 'Estimated Contour Area (in^2)':[], 'Time':[], 'Left Image':[], 'Middle Image':[], 'Right Image':[]}
        if saved == True:
            df_bats.loc[len(df_bats.index)] = ['bat', middle_id, left_center, middle_center, '-', depth, left_pre, middle_pre , '-', left_cnt, middle_cnt, '-', left_area, middle_area, '-', estimate_real_area, estimate_time, left_path, middle_path, '-']
    

    return df_bats