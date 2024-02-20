from smartbiofinder.utility.utils_cv import *
import cv2

def read_calibration_matrix():
    cv_file = cv2.FileStorage() #file must be in same folder as the videos
    cv_file.open('../calibration/stereoMap_16.xml', cv2.FileStorage_READ)

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


def match_frames():
    pass

