import sys
sys.path.append("..")
from os import listdir
from os.path import isfile, isdir, join
from myPackage.ROI import *
import numpy as np
import csv

import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

roi = ROI()

# 開啟 CSV 檔案
DLC_csv = np.loadtxt( open ( "DLC.csv" , encoding= 'big5' ), dtype=str , delimiter= ',' )
DLC_csv = DLC_csv.tolist()

# 以迴圈輸出每一列
for j in range(1,len(DLC_csv[0])):
    mydir = DLC_csv[0][j]
    files = listdir(mydir)
    files = natural_sort(files)
    print()
    print(mydir)
    DLC_csv[0].append(mydir)
    

    for i, f in enumerate(files):
        fullpath = join(mydir, f)
        print(f)
        
        roi.img = cv2.imread("{}/{}".format(mydir, f))
        roi.set_dxo_roi_img(TEST=False)
        # roi.set_gamma()
        # roi.roi_img = cv2.GaussianBlur(roi.roi_img,(3,3),0)
        # roi.roi_img = cv2.erode(roi.roi_img, kernel = np.ones((2, 2), np.uint8), iterations=1)
        # roi.roi_img = cv2.morphologyEx(roi.roi_img, cv2.MORPH_CLOSE, kernel = np.ones((3, 3)), iterations=1)
        # DLC_csv[i+1].append(roi.get_Imatest_any_sharp(roi.roi_img))
        
        DLC_csv[i+1].append(roi.get_average_gnorm())
        # print(DLC_csv[i+1][j])

with open("DLC_sharp.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(DLC_csv)
