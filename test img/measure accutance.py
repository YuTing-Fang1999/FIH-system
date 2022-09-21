import sys
sys.path.append("..")
from myPackage.ROI import *

roi = ROI()

# DXO_acutance
# roi.dxo_roi_img = cv2.imread("dead-leaves-crop.jpg")
# print(roi.get_DXO_acutance())

# roi.dxo_roi_img = cv2.imread("dead-leaves-crop-blur.jpg")
# print(roi.get_DXO_acutance())

# gamma_Laplacian
# roi.roi_img = cv2.imread("dead-leaves-crop.jpg")
# roi.roi_gamma = roi.get_gamma()
# print(roi.get_gamma_Laplacian())

# roi.roi_img = cv2.imread("dead-leaves-crop-blur.jpg")
# roi.roi_gamma = roi.get_gamma()
# print(roi.get_gamma_Laplacian())

roi.roi_img = cv2.imread("OPPO Find X2 DLC_D65_1000_crop_1.531.jpg")
roi.roi_gamma = roi.get_gamma()
print(roi.get_gamma_Laplacian())

roi.roi_img = cv2.imread("OPPO Reno4 Pro DLC_D65_1000_crop_1.773.jpg")
roi.roi_gamma = roi.get_gamma()
print(roi.get_gamma_Laplacian())

roi.roi_img = cv2.imread("OPPO Find X2 DLC_H_1_crop_1.090.jpg")
roi.roi_gamma = roi.get_gamma()
print(roi.get_gamma_Laplacian())

roi.roi_img = cv2.imread("OPPO Reno4 Pro DLC_H_1_crop_1.320.jpg")
roi.roi_gamma = roi.get_gamma()
print(roi.get_gamma_Laplacian())