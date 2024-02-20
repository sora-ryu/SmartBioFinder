'''
This script is to test 2D thermal bat tracking system

How to Run:
(tf_gpu02) python3.9 test_object_tracker.py ../videos/Middle\ 2023-07-06_21_50_10.mp4

Input: A thermal video to analyze
Output: Bat/bird/insect tracking result
'''
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from smartbiofinder.model.object_tracker import *


def main():
    ##########################################################################################################################
    # Set the thresholds for bat only (allowing the false positive)
    ##########################################################################################################################
    thresholds = {}
    thresholds['bat'] = set_classify_threshold('bat')
    thresholds['bird'] = set_classify_threshold('bird')

    ##########################################################################################################################
    # Load models for prediction
    ##########################################################################################################################
    binary_model = '../../models/BatFinder_Smart_Video_BioFilter.h5'        # Relevant path from save_video_directory folder
    multiclass_model = '../../models/vgg16_params_v1_best.hdf5'

    ##########################################################################################################################
    # Specify a path to the directory containing videos we want to analyze (capable of batch processing)
    ##########################################################################################################################
    video_directory = sys.argv[1]
    vid_file = video_directory.split('/')[-1]
    save_video_directory = os.path.abspath('../results/' + vid_file[:-4])
    set_directory('../results')
    set_directory(save_video_directory)

    insect_saving_directory_name = save_video_directory + '/insect_detected/'
    set_directory(insect_saving_directory_name)

    bat_saving_directory_name = save_video_directory + '/bat_detected/'
    set_directory(bat_saving_directory_name)
        
    bird_saving_directory_name = save_video_directory + '/bird_detected/'
    set_directory(bird_saving_directory_name)

    ##########################################################################################################################
    # Run through the videos to detect and memory-buffer a video-frame object for analysis
    ##########################################################################################################################
    startTime = time.time()             # Record the elapsed time for the process

    filename, _, extension = vid_file.partition('.')

    print("File name: ", vid_file)

    csv_header_list = ["X", "Y", "Width", "Height", "Object ID", "Analysis Date", "Analysis Time", "Object Name", "Current Frame", "Probability", "Center X", "Center Y", "Detected Contour Array(px)", "Area", "Video Start Timestamp", "Frame Timestamp", "File Path"]
    objects = ['bat', 'bird', 'insect']

    for object_type in objects:
        object_filename = save_video_directory + '/' + filename + '_' + str(object_type) + '.csv'
        write_csv(csv_header_list, object_filename)
    
    ##################################################################################
    # instantiating video capture object from which we will grab individual 
    # frames of the video as OpenCV objects (numpy arrays)
    cap = cv2.VideoCapture(video_directory)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)           # asf: 30.0 fps / mp4: Middle 30.030505776063908 Right 30.031325505233884

    ##########################################################################################################################
    # Setting up a video writer for making output videos
    ##########################################################################################################################
    fourcc = set_vid_codec(op_system=3)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter('../results/' + filename + '_DET.mp4', fourcc, fps, (width, height), isColor=True)

    # Initialize the object for every video
    tracker = ObjectTracker(video_directory, save_video_directory, binary_model, multiclass_model, thresholds, fps, filename)

    frame_num = 0
    while True:
        ret, frame = cap.read() 	# try to read each frame of the video object and return (ret) a
                                    # boolean of success (True or False) along with vid frame object
        if ret is False:
            break
        
        ##########################################################################################################################
        # Handle frame - find contours and run inference on each contour
        ##########################################################################################################################
        # print("frame: ", frame.shape[:2])        # (height, width) = (600, 800) -> verified for mp4, asf
        contours = tracker.get_contours(frame)     # Find all of the contours for current frame
        # print("contours: ", contours)            # [cnt, cnt, cnt, ...] cnt: moving single object (consists of multiple pixel arrays)
        frame_with_detection = tracker.handle_frame(frame, contours, frame_num)
        out.write(frame_with_detection)           
        frame_num += 1

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    
    endTime = time.time() 
    executionTime = endTime - startTime
    print('Execution time in seconds: ' + str(executionTime))


if __name__ == "__main__":
    main()