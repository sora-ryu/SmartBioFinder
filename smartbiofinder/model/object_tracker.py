import os
import io
from io import BytesIO
import numpy as np
import cv2
import imutils
from random import randint
import time
import re
import sys

from smartbiofinder.model.base_tracker import *
import csv
from datetime import date, datetime, timedelta
import pandas as pd

import keras
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img

import PIL.Image as im
import csv
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from smartbiofinder.utility.utils import *
from smartbiofinder.utility.utils_ml import prediction


class ObjectTracker(EuclideanDistTracker):      # Takes the bounding box of the objct and save them into one array
    def __init__(self, video_directory, save_video_directory, binary_model, multiclass_model, thresholds, fps, filename):
        super().__init__()
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history = 3, varThreshold = 85, detectShadows=False)
        self.vid_path = video_directory
        self.save_vid_path = save_video_directory
        os.chdir(self.save_vid_path)

        # print(os.path.abspath(os.getcwd()))  # Get the current working directory path
        self.binary_model_object = load_model(binary_model)
        self.multiclass_model_object = load_model(multiclass_model)
        self.classes = {0: 'bat', 1: 'bird', 2: 'empty', 3: 'insect'}
        self.thresholds = thresholds
        self.fps = fps
        self.filename = filename
        self.colors = {'bat': (255,0,0), 'bird': (0,255,0), 'insect':(0,0,255)}     # BGR
    
    
    def get_contours(self, frame):

        roi = frame.copy()
        self.fgmask = self.object_detector.apply(frame)
        # Remove the noise from the background
        kernel = np.ones((3,3), np.uint8)
        self.fgmask = cv2.morphologyEx(self.fgmask, cv2.MORPH_OPEN, kernel, iterations = 1)   # Remove noise (erosion then dilation)
        self.fgmask = cv2.morphologyEx(self.fgmask, cv2.MORPH_CLOSE, kernel, iterations = 1)  # Close small holes inside the foreground objects (dilation then erosion)
        contours, _ = cv2.findContours(self.fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    

    def run_detection_models(self, image_of_detected_object_roi):

        BGR_roi = cv2.cvtColor(image_of_detected_object_roi, cv2.COLOR_BGR2RGB)
        image_of_detected_object_roi_image = im.fromarray(BGR_roi)
        image_of_detected_object_roi_image.save("buf_image.jpeg")

        # Here we do the prediction with binary model
        binary_pred = prediction("buf_image.jpeg", self.binary_model_object)       # confidence level number between [0,1] where 0: empty / 1: biological object
        if binary_pred[0][0] < 0.5:         # Skip if the probability is below the threshold
            return 'empty', None

        # Here we do the prediction with multi-class model
        multiclass_pred = prediction("buf_image.jpeg", self.multiclass_model_object)
        
        '''
        # Set the object as the one that has the maximum probability
        object_arg = np.argmax(multiclass_pred)
        object_type = self.classes[object_arg]          # classes = {0: 'bat', 1: 'bird', 2: 'empty', 3: 'insect'}
        print("[frame_num] object type: ", frame_num, object_type)
        object_prob = np.amax(multiclass_pred).round(2)
        '''

        # Set the object as bat if it has larger probability than the bat threshold
        # Else, set as the one that has the maximum probability over three categories - bird, empty, insect
        if multiclass_pred[0][0] > self.thresholds['bat']:
            object_arg = 0      # bat class index
        else:
            object_arg = np.argmax(multiclass_pred[0][1:])
        
        if object_arg == 1 and multiclass_pred[0][1] < self.thresholds['bird']:     # Remove false postive birds
            object_arg = 2
        
        object_type = self.classes[object_arg]

        return object_type, multiclass_pred
    

    def handle_frame(self, frame, contours, frame_num):
        
        # detections = [[] for _ in range(len(self.classes))]
        org_frame = frame.copy()

        # Go through every contours
        for cnt in contours:
            # Calculate area and remove small elements
            M = cv2.moments(cnt)
            area = M["m00"]                         # Count all non-zero pixels. Could be calculated by cv2.contourArea(cnt) as well.
            
            if area > 40:                   # remove too small contours to identify
                cX = int(M["m10"]/area)     # calculate the center of the contour object along the X (horizontal pixel row) axis of video frame            				
                cY = int(M["m01"]/area)
                
                x, y, w, h = cv2.boundingRect(cnt)          # left, top, width, height
                image_of_detected_object_roi = frame[cY - 25 : cY + 25, cX - 25 : cX + 25].copy()     # input dimension should be (50,50)
                                                                                                    # Design choice: remove contours which area > 2500? Just focus on rotor swept area.. not close objects
                
                if cY < 45 or cY > 575 or cX < 25 or cX > 775:  # Locates on the black bar at the top of the screen (Should be 20<Cy-25, Cy+25<600, 0<Cx-25, Cx+25<800)
                    continue
                
                object_type, multiclass_pred = self.run_detection_models(image_of_detected_object_roi)
                if object_type == 'empty':
                    continue
                
                print(f"[Frame #{frame_num}] object type: {object_type}")
                object_id = self.update(object_type, cX, cY, frame_num)
                cv2.rectangle(frame, (cX-25, cY-25), (cX+25, cY+25), self.colors[object_type], 2)
                cv2.putText(frame, f'Area: {area} {object_type} {object_id}', (cX, cY+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[object_type], 2)

                # Dump into csv files
                os.chdir(os.path.join(self.save_vid_path, str(object_type) + '_detected/'))
                parent_directory = self.filename + '/' + str(object_type) + '_detected/'
                id_directory = str(object_type) + '_id_' + str(object_id)
                set_directory(id_directory)
                new_object_directory_name = parent_directory + id_directory
                self.save_tracking_results([x,y,w,h], object_id, object_type, frame_num, multiclass_pred[0], cX, cY, cnt, area, new_object_directory_name)
                
                # Save frame and roi
                cv2.imwrite(str(object_type) + '_detected/' + id_directory + '/frame_num_' + str(frame_num) + '_bbox.jpg', frame)
                cv2.imwrite(str(object_type) + '_detected/' + id_directory + '/frame_num_' + str(frame_num) + '.jpg', org_frame)
                cv2.imwrite(str(object_type) + '_detected/' + id_directory + '/frame_num_' + str(frame_num) + '_zoom_in1.jpg', image_of_detected_object_roi)   # added for zoom-in

        
        return frame


    def save_tracking_results(self, bounding_box, object_id, object_type, frame_num, multiclass_pred, cX, cY, cnt, area, new_object_directory_name):

        # Analysis time
        timestamp = datetime.now()
        day = timestamp.strftime("%Y/%m/%d")
        ptime = timestamp.strftime("%H:%M:%S.%f")

        x, y, w, h = bounding_box
        box_id = [x, y, w, h, object_id, day, ptime, object_type, frame_num, multiclass_pred, cX, cY, cnt, area]

        os.chdir(self.save_vid_path)
        object_filename = self.filename + '_' + str(object_type) + '.csv'
        start_timestamp = datetime(*map(int, re.split('-|_', self.filename.split(' ')[-1])))
        video_type = self.filename.split(' ')[0]
        time_offset = {'Middle': 0, 'Right': 0, 'Left': 0}    # Delay occurs
        frame_timestamp = start_timestamp + timedelta(seconds=frame_num/self.fps + time_offset[video_type])     # Add time offset depending on video type
        file_path = os.path.join(new_object_directory_name, 'frame_num_' + str(frame_num) + '.jpg')
        csv_elements = box_id + [start_timestamp.strftime("%Y/%m/%d %H:%M:%S.%f"), frame_timestamp.strftime("%Y/%m/%d %H:%M:%S.%f"), file_path]
        write_csv(csv_elements, object_filename)

