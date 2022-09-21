import sys
sys.path.append("..")
from myPackage.ROI import *

roi = ROI()

# crop
# roi.roi_img = cv2.imread("OPPO Find X2 DLC_D65_1000_crop_1.531.jpg")
# roi.roi_gamma = roi.get_gamma()
# print(roi.get_gamma_Laplacian())

# roi.roi_img = cv2.imread("OPPO Reno4 Pro DLC_D65_1000_crop_1.773.jpg")
# roi.roi_gamma = roi.get_gamma()
# print(roi.get_gamma_Laplacian())

# roi.roi_img = cv2.imread("OPPO Find X2 DLC_H_1_crop_1.090.jpg")
# roi.roi_gamma = roi.get_gamma()
# print(roi.get_gamma_Laplacian())

# roi.roi_img = cv2.imread("OPPO Reno4 Pro DLC_H_1_crop_1.320.jpg")
# roi.roi_gamma = roi.get_gamma()
# print(roi.get_gamma_Laplacian())


roi.img = cv2.imread("OPPO Find X2 DLC/A_5_1.246.jpg")
roi.set_dxo_roi_img(TEST=False)
# roi.set_gamma()
print(roi.get_Laplacian_sharpness())

roi.img = cv2.imread("OPPO Reno4 Pro DLC/A_5_1.237.jpg")
roi.set_dxo_roi_img(TEST=False)
# roi.set_gamma()
print(roi.get_Laplacian_sharpness())


roi.img = cv2.imread("OPPO Find X2 DLC/A_20_1.290.jpg")
roi.set_dxo_roi_img(TEST=False)
# roi.set_gamma()
print(roi.get_Laplacian_sharpness())

roi.img = cv2.imread("OPPO Reno4 Pro DLC/A_20_1.241.jpg")
roi.set_dxo_roi_img(TEST=False)
# roi.set_gamma()
print(roi.get_Laplacian_sharpness())