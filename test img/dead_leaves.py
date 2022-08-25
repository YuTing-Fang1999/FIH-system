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

    edged = cv2.Canny(resize_gray_im, 300, 600)
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
    marker_angle = 0

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
            if find and np.linalg.norm(np.array(coor[-1])-np.array([r,c]))<5:
                continue

            if not find:
                if angle>100: 
                    not_right_angle = True
                find = True
                marker_angle = angle
                

            coor.append((r,c)) # row, col

    coor = np.array(coor)
    scale = im.shape[0]/resize_im.shape[0]
    print(scale)
    coor = np.around(coor * scale).astype(int)


    # find center
    vec = coor[0] - coor[3] # → ↓
    mid = coor[3] + vec/2
    mid = np.around(mid).astype(int)

    len = np.linalg.norm(vec)

    # 由下到上，右到左
    for c in coor:
        # 在中心點畫上黃色實心圓
        cv2.circle(im, (c[1], c[0]), int(len/300), (1, 227, 254), -1)
    # cv2.imshow("im", ResizeWithAspectRatio(im, height=600))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(coor)

    topLeft = np.around(coor[3] + np.array([-1,-0.9])*len*0.07).astype(int)
    bottomRight = np.around(coor[0] + np.array([1,0.9])*len*0.07).astype(int)
    roi_img = im[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]

    print(marker_angle)

    if not_right_angle: 
        roi_img=rotate_img(roi_img,180)


    # cv2.imshow("roi_img", ResizeWithAspectRatio(roi_img, height=600))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return roi_img, coor-topLeft
    
def get_roi_region(im, coor):

    # find center
    vec = coor[0] - coor[3] # → ↓
    mid = coor[3] + vec/2
    mid = np.around(mid).astype(int)

    len = np.linalg.norm(vec)

    

    # find center
    vec = coor[0] - coor[3] # → ↓
    mid = coor[3] + vec/2
    mid = np.around(mid).astype(int)

    len = np.linalg.norm(vec)

    # 由下到上，右到左
    for c in coor:
        # 在中心點畫上黃色實心圓
        cv2.circle(im, (c[1], c[0]), int(len/300), (1, 227, 254), -1)
        # cv2.putText(im, "({}, {})".format(c[0], c[1]), (c[1]-30, c[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    direction = [[-1,0], [0,1],[1,0],[0,-1]]

    mean_value=[]
    # if is_gray_value: gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    
    for i, d in enumerate(direction):
        rate = len*0.27
        vec = np.array(d)*rate
        local_mid = np.around(mid + vec).astype(int)
        
        rec_roi = get_rec_roi(im, local_mid, len*0.03).reshape(-1,3).mean(axis=0)
        mean_value.append(rec_roi)

        cv2.circle(im, (local_mid[1], local_mid[0]), int(len/300), (1, 227, 254), -1)
        cv2.putText(im, "{}".format(np.around(rec_roi).astype(int)), (local_mid[1]-int(len/20), local_mid[0]-int(len/50)), cv2.FONT_HERSHEY_SIMPLEX, len/2000, (0, 255, 0), int(len/500), cv2.LINE_AA)

        # cv2.imshow("roi", rec_roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if i%2: local_d2 = [[-1,0],[1,0]]
        else: local_d2 = [[0,-1],[0,1]]
        
        for d2 in local_d2:
            rate = len*0.09
            vec = np.array(d2)*rate
            p = np.around(local_mid + vec).astype(int)
            
            rec_roi = get_rec_roi(im, p, len*0.03).reshape(-1,3).mean(axis=0)
            mean_value.append(rec_roi)
            # print(rec_roi.shape)

            cv2.circle(im, (p[1], p[0]), int(len/300), (1, 227, 254), -1)
            cv2.putText(im, "{}".format(np.around(rec_roi).astype(int)), (p[1]-int(len/20), p[0]+int(len/50)), cv2.FONT_HERSHEY_SIMPLEX, len/2000, (0, 255, 0), int(len/500), cv2.LINE_AA)

    cv2.imshow("roi", ResizeWithAspectRatio(im, height=600))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(np.array(mean_value).T.shape)
    return np.sort(np.array(mean_value).T)/255

def func(x, a, b, c):
    return a*(np.array(x)**b) + c

def invert_OECF(roi_img, OECF_patch, color_name):
    x=np.linspace(0, 1, num=12)
    popt, pcov = curve_fit(func, x, OECF_patch)
    a,b,c=np.around(popt,3)

    y = a*(x**b) + c
    inverse_y = ((y-c)/a)**(1/b)

    plt.plot(x, OECF_patch, 'r', label=color_name)
    plt.plot(x, y, 'c', label="y = {} * (x**{}) + {} (approximate {})".format(a,b,c,color_name))
    plt.plot(x, inverse_y, 'k', label="inverse_y")
    plt.plot(x, inverse_y**(1/2.2), 'm', label="inverse_y gamma=(1/2.2)")
    plt.legend()
    plt.show()

    inverse_y = ((OECF_patch-c)/a)**(1/b)
    plt.plot(x, r, 'r', label=color_name)
    plt.plot(x, y, 'c', label="y = {} * (x**{}) + {} (approximate {})".format(a,b,c,color_name))
    plt.plot(x, inverse_y, 'k', label="inverse_y")
    plt.plot(x, inverse_y**(1/2.2), 'm', label="inverse_y gamma=(1/2.2)")
    plt.legend()
    plt.show()


# im=cv2.imread('OPPO Find X2 DLC/A_5.jpg')
# im=cv2.imread('OPPO Find X2 DLC/A_20.jpg')
# im=cv2.imread('OPPO Find X2 DLC/A_100.jpg')
# im=cv2.imread('OPPO Find X2 DLC/A_300.jpg')
# im=cv2.imread('OPPO Find X2 DLC/D65_1000.jpg')
im=cv2.imread('OPPO Find X2 DLC/H_1.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_20.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_100.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_300.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_1000.jpg')
# im=cv2.imread('dead-leaves-target.jpg')

roi_img, coor = get_roi_img_and_coor(im)
r, g, b = get_roi_region(roi_img, coor)

invert_OECF(roi_img, OECF_patch=r, color_name="r channel")
# plt.plot(x, g, 'g', label="g chanel")
# plt.plot(x, b, 'b', label="b chanel")

# plt.plot(x, gray_value1, 'r', label="mean of 12 patch of the H_1.jpg")
# plt.plot(x, gray_value2, 'g')
# plt.plot(x, gray_value3, 'b')
# plt.plot(x, x**(0.45), 'm', label="gamma = 0.45 (1/2.2)")
# plt.plot(x, x**(popt[0]), 'c', label="gamma = {}".format(popt[0]))
# plt.legend()
# plt.show()

# cv2.imshow("roi", ResizeWithAspectRatio(im, height=1000))
# cv2.waitKey(0)
# cv2.destroyAllWindows()













