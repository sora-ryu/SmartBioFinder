import tkinter as tk
from tkinter import filedialog
import csv
import keras
import os
from PIL import Image
import cv2
import pandas as pd


def get_file():
    root = tk.Tk()
    root_path = filedialog.askopenfilename()
    root.withdraw()
    return root_path

def get_directory():
    root = tk.Tk()
    root_path = filedialog.askdirectory()
    root.withdraw()
    return root_path

def set_directory(dir_name):
    if os.path.isdir(dir_name) == False:
        os.mkdir(dir_name)

def set_detection_threshold():
    enter_val = float(input('Please enter an object-detection probability threshold from 0-100: ') or 50)       # default: 30
    pct_val = enter_val * 0.01
    
    return pct_val

def set_classify_threshold(animal):
    '''
    if animal == 'bat':
        enter_val = float(input('Please enter ' + animal + '-classification probability threshold from 0-100: ') or 10)    # default: 10
    elif animal == 'bird':
        enter_val = float(input('Please enter ' + animal + '-classification probability threshold from 0-100: ') or 85)    # default: 85
    elif animal == 'insect':
        enter_val = float(input('Please enter ' + animal + '-classification probability threshold from 0-100: ') or 85)    # default: 85
    else:
        enter_val = float(input('Please enter ' + animal + '-classification probability threshold from 0-100: ') or 30)    # default: 30
    '''
    if animal == 'bat':
        enter_val = 10           # For prev_low_cnt datasets: bat threshold = 30
    
    elif animal == 'bird':
        enter_val = 85
    
    pct_val = enter_val * 0.01
    
    return pct_val

def set_vid_codec(op_system=3):
    # Here we set up a video compression codec, which can be computing platform specific.
    if op_system == 1:    # Windows
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    elif op_system == 2:   # MacOS
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    else:                  # Linux and everything else
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    return fourcc

        
def write_csv(data,filename):
    with open(filename, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

def check_images_size(data_folder):

    data_info = {}
    
    for root, dirs, files in os.walk(data_folder):

        # First level
        for dirname in dirs:
            data_info[dirname] = set()

        count, total = 0, 0

        # Second level
        dir_name = str(root).split('/')[-1]
        for filename in files:
            img_path = os.path.join(root, filename)
            with Image.open(img_path) as img:
                if not img.size in data_info[dir_name]:
                    data_info[dir_name].add(img.size)
                if img.size != (50, 50):
                    print('Delete %s : %s' % (img_path, img.size))
                    # os.remove(img_path)     # delete unmatched image
                    count += 1
                total += 1
        
        print(dir_name, total)

        if count == 0:
            continue
        
        print("The number of non 50 X 50 images under %s category: %s => %s / %s => %s" % (dir_name, count, count, total, count/total))

    
    return data_info


def read_csv_file(left_filepath, middle_filepath, right_filepath):

    # Read dataframes (2D Bat Tracking Output from each video)
    df_left = pd.read_csv(left_filepath)
    df_middle = pd.read_csv(middle_filepath)
    df_right = pd.read_csv(right_filepath)
    
    left_file, separator, extension = left_filepath.split("/")[-1].partition('.')
    middle_file, separator, extension = middle_filepath.split("/")[-1].partition('.')
    right_file, separator, extension = right_filepath.split("/")[-1].partition('.')

    return df_left, df_middle, df_right, left_file, middle_file, right_file

