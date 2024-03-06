import numpy as np

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

