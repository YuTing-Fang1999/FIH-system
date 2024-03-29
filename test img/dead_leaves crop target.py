import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
# PYTHON:数据拟合求解方程参数
from scipy.optimize import curve_fit

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        # if h<height: return image
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # if w<width: return image
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def rotate_img(img, angle):
    (h, w, d) = img.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))
    
    return rotate_img

def get_rec_roi(im, p, w):
    topLeft = p + np.around(np.array([-1,-1])*w).astype(int)
    bottomRight = p + np.around(np.array([1,1])*w).astype(int)
    rec_roi = im[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1],:].copy()
    cv2.rectangle(im, (topLeft[1], topLeft[0]), (bottomRight[1], bottomRight[0]), (255,0,0), int(w/30))
    return rec_roi

def get_roi_img_and_coor(im):
    resize_im = ResizeWithAspectRatio(im, height=800)
    resize_gray_im = cv2.cvtColor(resize_im, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("resize_im", resize_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edged = cv2.Canny(resize_gray_im, 300, 550)
    # cv2.imshow("edged", ResizeWithAspectRatio(edged, height=800))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((1,2), np.uint8) 
    edged = cv2.dilate(edged, kernel, iterations = 1)
    # cv2.imshow("dilate", ResizeWithAspectRatio(edged, height=800))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    backgroundSkeleton = skeletonize(np.where(edged==255,1,0))
    backgroundSkeleton = np.where(backgroundSkeleton==1,255,0).astype('uint8')  
    # cv2.imshow("backgroundSkeleton", ResizeWithAspectRatio(backgroundSkeleton, height=800))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cnts, _ = cv2.findContours(backgroundSkeleton.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    coor = []
    
    # 依次處理每個Contours

    find = False
    not_right_angle = False

    for c in cnts:

        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area)/hull_area

        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h

        if np.around(solidity, 1) == 0.6 and np.around(aspect_ratio, 1) == 1:

            (c,r),(MA,ma),angle = cv2.fitEllipse(c)
            # 往右下遞增
            r, c = int(r), int(c)

            # 避免重複尋找
            if find and np.linalg.norm(np.array(coor[-1][:2])-np.array([r,c]))<5:
                continue

            if not find:
                find = True

            coor.append((r,c,angle)) # row, col


    assert len(coor) == 4
    coor = sorted(coor, key=lambda x: x[0], reverse=True)
    coor[:2] = sorted(coor[:2], key=lambda x: x[1], reverse=True)
    coor[2:] = sorted(coor[2:], key=lambda x: x[1], reverse=True)
    if coor[0][2]<100: not_right_angle = True
    
    coor = np.array(coor)
    coor = coor[:,:2]

    scale = im.shape[0]/resize_im.shape[0]
    coor = np.around(coor * scale).astype(int)

    # find center
    vec = coor[0] - coor[3] # → ↓
    mid = coor[3] + vec/2
    mid = np.around(mid).astype(int)

    length = np.linalg.norm(vec)

    # 由下到上，右到左
    for c in coor:
        # 在中心點畫上黃色實心圓
        cv2.circle(im, (c[1], c[0]), int(length/300), (1, 227, 254), -1)
        # cv2.imshow("im", ResizeWithAspectRatio(im, height=600))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    topLeft = np.around(coor[3] + np.array([-1,-0.9])*length*0.07).astype(int)
    bottomRight = np.around(coor[0] + np.array([1,0.9])*length*0.07).astype(int)
    roi_img = im[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]

    if not_right_angle: 
        print('not right angle, rotate 180')
        roi_img=rotate_img(roi_img,180)


    # cv2.imshow("roi_img", ResizeWithAspectRatio(roi_img, height=600))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return roi_img, coor-topLeft
    
def get_roi_region(im, coor, file_name):

    # find center
    vec = coor[0] - coor[3] # → ↓
    mid = coor[3] + vec/2
    mid = np.around(mid).astype(int)
    length = np.linalg.norm(vec)

    # find target ROI
    rate = np.linalg.norm(vec)*0.2
    vec = np.array([1,1])*rate
    topLeft = np.around(mid - vec).astype(int)
    bottomRight = np.around(mid + vec).astype(int)
    dxo_roi_img = im[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]
    # cv2.imwrite(file_name.replace('/','_').split('.')[0]+'_crop.jpg', dxo_roi_img)

    # 繪製方框
    # cv2.rectangle(im, (topLeft[1], topLeft[0]), (bottomRight[1], bottomRight[0]), (255,0,0), 10)
    # cv2.imshow("roi_img", ResizeWithAspectRatio(im, height=800))
    # cv2.waitKey(100)

    # 由下到上，右到左
    for c in coor:
        # 在中心點畫上黃色實心圓
        cv2.circle(im, (c[1], c[0]), int(length/300), (1, 227, 254), -1)
        # cv2.putText(im, "({}, {})".format(c[0], c[1]), (c[1]-30, c[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    direction = [[-1,0], [0,1],[1,0],[0,-1]]

    OECF_patch=[]
    # if is_gray_value: gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    
    for i, d in enumerate(direction):
        rate = length*0.27
        vec = np.array(d)*rate
        local_mid = np.around(mid + vec).astype(int)
        
        rec_roi = get_rec_roi(im, local_mid, length*0.03)
        OECF_patch.append(rec_roi)

        cv2.circle(im, (local_mid[1], local_mid[0]), int(length/300), (1, 227, 254), -1)
        cv2.putText(im, "{}".format(np.around(rec_roi).reshape(-1,3).mean(axis=0).astype(int)), (local_mid[1]-int(length/20), local_mid[0]-int(length/50)), cv2.FONT_HERSHEY_SIMPLEX, length/2000, (255, 0, 0), int(length/500), cv2.LINE_AA)

        # cv2.imshow("roi", rec_roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if i%2: local_d2 = [[-1,0],[1,0]]
        else: local_d2 = [[0,-1],[0,1]]
        
        for d2 in local_d2:
            rate = length*0.09
            vec = np.array(d2)*rate
            p = np.around(local_mid + vec).astype(int)
            
            rec_roi = get_rec_roi(im, p, length*0.03)
            OECF_patch.append(rec_roi)
            # print(rec_roi.shape)

            cv2.circle(im, (p[1], p[0]), int(length/300), (1, 227, 254), -1)
            cv2.putText(im, "{}".format(np.around(rec_roi).reshape(-1,3).mean(axis=0).astype(int)), (p[1]-int(length/20), p[0]+int(length/50)), cv2.FONT_HERSHEY_SIMPLEX, length/2000, (255, 0, 0), int(length/500), cv2.LINE_AA)

    cv2.imshow("roi", ResizeWithAspectRatio(im, height=600))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(OECF_patch)

def cal_mean_OECF_patch(OECF_patch):
    mean_value = OECF_patch.reshape(12,-1,3).mean(axis=1)
    # print(mean_value.shape)
    return np.sort(np.array(mean_value).T)/255

def get_OECF_patch_mean(file_name):
    im=cv2.imread(file_name)
    roi_img, coor = get_roi_img_and_coor(im.copy())
    OECF_patch = get_roi_region(roi_img.copy(), coor, file_name)
    return cal_mean_OECF_patch(OECF_patch)

def func(x, a, b):
    return x**b

# file_name1 = "OPPO Find X2 DLC/H_1.jpg"
# file_name2 = "OPPO Find X2 DLC/D65_1000.jpg"
# file_name1 = "OPPO Reno4 Pro DLC/H_1.jpg"
file_name1 = "OPPO Reno4 Pro DLC/D65_1000.jpg"
file_name2 = "dead-leaves-target.jpg"

b1, g1, r1 = get_OECF_patch_mean(file_name1)
b2, g2, r2 = get_OECF_patch_mean(file_name2)
# b3, g3, r3 = get_OECF_patch_mean(file_name3)


x=np.linspace(0, 1, num=12)
popt, pcov = curve_fit(func, x, b2)

plt.plot(x, b1, 'r', label=file_name1)
plt.plot(x, b2, 'c', label=file_name2)
plt.plot(x, popt[0]*x**(popt[1]), 'm', label="{}*x**{}".format(popt[0], popt[1]))
# plt.plot(b2, b1, 'k')
plt.legend()
plt.show()













