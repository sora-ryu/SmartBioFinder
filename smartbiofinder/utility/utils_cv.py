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
    baseline = 95.25          #Distance between the cameras [cm] -> need to verify
    f = 19             #Camera lense's focal length [mm]
    alpha = 19.4       #Camera field of view in the horizontal plane [degrees]
    mean = 173.1121542
    std = 34.94402137
    normalized = 0.876034097 #This was found at 255 feet in the truck example

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    width_right = 640
    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    x_rightd = right_point[0]
    x_leftd = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_leftd-x_rightd      #Displacement between left and right frames [pixels]
    standardized_disparity = (disparity-mean)/std
    normalized_disparity = standardized_disparity + normalized
    
    # CALCULATE DEPTH Z:
    zDepth = ((baseline * f_pixel) / disparity) * 0.032808          # cm -> feet (by multiplying 0.032808)
    linearized_depth = (0.5877 + normalized_disparity)/0.0213       # feet

    return abs(zDepth), abs(linearized_depth)