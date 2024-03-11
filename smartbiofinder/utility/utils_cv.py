import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def splitByIndentation(data : str):
    indented = False
    output = []
    prevLine = ''
    # remove empty lines
    lines = [line for line in data.split('\n') if line.strip() != '']
    print('lines:\n', lines)

    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            if not indented:
                output.append([prevLine])
                indented = True
            output[-1].append(line.strip()) # append to last block and remove whitespace
        else:
            prevLine = line
            indented = False
    return output


def read_cnt_from_csv(cnt):
    # Need to clean up!!
    a = cnt.replace("[[ ","")
    b = a.replace("]]\n\n "," ")
    c = b.replace("]]]","")
    d = c.replace("[","")
    e = d.split(" ")
    f = []
    for ele in e:
        if ele.strip():
            f.append(ele)
    lst = []
    for i in range(len(f)-1):
        if (i % 2) == 0:
            if '...' in f[i]:       # There's some cases where contours array is so long that it just skipped middle as '...'
                return np.array([])
            n = int(f[i])
            m = int(f[i+1])
            lst.append([n,m])
        else:
            continue
    ray = np.array(lst)

    return ray


def create_df_analysis():
    # Dataframe to save the results
    headers = {'Object Type':[], 'Object ID':[], 'Rectified Left Center (px)':[], 'Rectified Middle Center (px)':[], 'Rectified Right Center (px)':[], 'Depth Prediction (cm)':[], 'Left Multi Predictions':[], 'Middle Multi Predictions':[], 'Right Multi Predictions':[],  'Left Detected Countour Array (px)':[], 'Middle Detected Countour Array (px)':[], 'Right Detected Countour Array (px)':[], 'Left Contour Area (px)':[], 'Middle Contour Area (px)':[], 'Right Contour Area (px)':[], 'Estimated Contour Area (in^2)':[], 'Time':[], 'Left Image':[], 'Middle Image':[], 'Right Image':[]}
    df_bats = pd.DataFrame(headers)

    return df_bats