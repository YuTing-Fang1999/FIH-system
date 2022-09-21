from tkinter.messagebox import NO
import cv2
import numpy as np
# from scipy.signal import convolve2d
from skimage.morphology import skeletonize
import math
from math import e
import random


class ROI:
    # 建構式
    def __init__(self):
        self.img = None
        self.roi_img = None
        self.roi_gamma = None
        self.roi_coordinate = None

    def set_roi_img(self, img, roi_coordinate):
        self.img = img
        self.roi_coordinate = roi_coordinate
        coor = self.roi_coordinate
        self.roi_img = self.img[int(coor.r1):int(
            coor.r2), int(coor.c1):int(coor.c2), :]
        self.roi_gamma = self.get_gamma()

    ###### sharpness ######
    def get_gamma(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gamma = 0.5
        invGamma = 1.0 / gamma
        I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized
        return I

    def get_sharpness(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        return np.round(np.sqrt(cv2.Laplacian(I, cv2.CV_64F).var()), 4)

    def get_chroma_noise(self):
        I = self.roi_img.copy()
        img_yuv = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        return u.std()+v.std()

    def get_noise(self):
        # Reference: J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996 [PDF]
        # https://loli.github.io/medpy/generated/medpy.filter.noise.immerkaer.html
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        H, W = I.shape

        M = np.array(
            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
            )

        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(I, cv2.CV_64F, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

        return np.round(sigma, 4)

    def get_average_gnorm(self):
        # https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gy, gx = np.gradient(I)
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        return np.round(sharpness, 4)

    def get_Imatest(self):
        I = self.roi_gamma.copy()
        sobelx = cv2.Sobel(I, cv2.CV_16S, 1, 0)  # x方向梯度 ksize默認為3x3
        sobely = cv2.Sobel(I, cv2.CV_16S, 0, 1)  # y方向梯度

        abx = np.abs(sobelx)  # 取sobelx絕對值
        aby = np.abs(sobely)  # 取sobely絕對值

        sx = np.mean(abx)/np.mean(I)
        sy = np.mean(aby)/np.mean(I)

        sTotal = np.sqrt(sx*sx + sy*sy)

        return np.round(sTotal, 4)

    def get_gamma_Sobel(self):
        I = self.roi_gamma.copy()

        sobelx = cv2.Sobel(I, cv2.CV_16S, 1, 0)  # x方向梯度 ksize默認為3x3
        sobely = cv2.Sobel(I, cv2.CV_16S, 0, 1)  # y方向梯度

        var = np.mean(sobelx**2 + sobely**2)

        return np.round(np.sqrt(var), 4)

    def get_gamma_Laplacian(self):
        I = self.roi_gamma.copy()
        return np.round(np.sqrt(cv2.Laplacian(I, cv2.CV_64F, ksize=1).var()), 4)

    def get_Imatest_any_sharp(self):
        I = self.roi_gamma.copy()
        ### old ###
        # gy, gx = np.gradient(I)

        # self.H = gx.std()
        # self.V = gy.std()
        ######

        sobelx = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=3)
        self.H = np.mean(np.abs(sobelx))/np.mean(I) # 官網公式
        self.V = np.mean(np.abs(sobely))/np.mean(I) # 官網公式

        self.H *= 100 #百分比
        self.V *= 100 

        return np.round(((self.H**2 + self.V**2)/2)**(0.5), 4)

    def get_H(self):
        return np.round(self.H, 4)

    def get_V(self):
        return np.round(self.V, 4)

    ############ DXO deadleaves #############
    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def rotate_img(self, img, angle):
        (h, w, d) = img.shape # 讀取圖片大小
        center = (w // 2, h // 2) # 找到圖片中心
        
        # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 第三個參數變化後的圖片大小
        rotate_img = cv2.warpAffine(img, M, (w, h))
        
        return rotate_img

    def get_rec_roi(self, im, p, w):
        topLeft = p + np.around(np.array([-1,-1])*w).astype(int)
        bottomRight = p + np.around(np.array([1,1])*w).astype(int)
        rec_roi = im[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1],:].copy()
        cv2.rectangle(im, (topLeft[1], topLeft[0]), (bottomRight[1], bottomRight[0]), (255,0,0), int(w/30))
        return rec_roi

    def get_roi_img_and_coor(self, im):
        resize_im = self.ResizeWithAspectRatio(im, height=800)
        resize_gray_im = cv2.cvtColor(resize_im, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("resize_im", resize_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        edged = cv2.Canny(resize_gray_im, 350, 550)
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
        cv2.imshow("backgroundSkeleton", self.ResizeWithAspectRatio(backgroundSkeleton, height=800))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                if find and np.linalg.norm(np.array(coor[-1][:2])-np.array([r,c]))<5:
                    continue

                if not find:
                    find = True
                    marker_angle = angle
                    

                coor.append((r,c,angle)) # row, col

        print(len(coor))
        print(coor)
        assert len(coor) == 4
        coor = sorted(coor, key=lambda x: x[0], reverse=True)
        coor[:2] = sorted(coor[:2], key=lambda x: x[1], reverse=True)
        coor[2:] = sorted(coor[2:], key=lambda x: x[1], reverse=True)
        print(coor)
        if coor[0][2]<100: not_right_angle = True
        
        coor = np.array(coor)
        coor = coor[:,:2]
        print(coor)

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

        print(marker_angle)

        if not_right_angle: 
            print('not right angle, rotate 180')
            roi_img=self.rotate_img(roi_img,180)


        cv2.imshow("circle", self.ResizeWithAspectRatio(roi_img, height=600))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return roi_img, coor-topLeft

    def get_roi_region(self, im, coor, file_name):
   
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
        self.dxo_roi_img = dxo_roi_img
        self.roi_img = cv2.cvtColor(self.dxo_roi_img, cv2.COLOR_BGR2GRAY)

        # 繪製方框
        cv2.rectangle(im, (topLeft[1], topLeft[0]), (bottomRight[1], bottomRight[0]), (255,0,0), 10)

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
            
            rec_roi = self.get_rec_roi(im, local_mid, length*0.03)
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
                
                rec_roi = self.get_rec_roi(im, p, length*0.03)
                OECF_patch.append(rec_roi)
                # print(rec_roi.shape)

                cv2.circle(im, (p[1], p[0]), int(length/300), (1, 227, 254), -1)
                cv2.putText(im, "{}".format(np.around(rec_roi).reshape(-1,3).mean(axis=0).astype(int)), (p[1]-int(length/20), p[0]+int(length/50)), cv2.FONT_HERSHEY_SIMPLEX, length/2000, (255, 0, 0), int(length/500), cv2.LINE_AA)

        cv2.imshow("roi", self.ResizeWithAspectRatio(im, height=600))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return np.array(OECF_patch)

    def cal_mean_OECF_patch(self, OECF_patch):
        mean_value = OECF_patch.reshape(12,-1,3).mean(axis=1)
        print(mean_value.shape)
        return np.sort(np.array(mean_value).T)/255

    def set_dxo_roi_img(self):
        roi_img, coor = self.get_roi_img_and_coor(self.img.copy())
        OECF_patch = self.get_roi_region(roi_img.copy(), coor, "")
        return self.cal_mean_OECF_patch(OECF_patch)

    # compute the average of over all directions
    def radialAverage(self, arr):
        assert arr.shape[0] == arr.shape[1]

        N = arr.shape[0]
        # Calculate the indices from the image
        y, x = np.indices(arr.shape)
        center = np.array([N//2, N//2])
        r = np.hypot(x - center[0], y - center[1])

        # 依半徑大小將 r 的 index 由小排到大
        ind = np.argsort(r.flat)
        # 依 index 取 r (由小排到大的半徑)
        r_sorted = r.flat[ind]
        # 依 index 取 img 的值
        i_sorted = arr.flat[ind]

        # 將 r 轉為整數
        r_int = r_sorted.astype(int)

        # 找出半徑改變的位置 rind=[0,8] 代表在0~1、8~9之間改變 => 0, 1~8, 9~24
        deltar = r_int - np.roll(r_int, -1)  # shift and substract
        rind = np.where(deltar != 0)[0]       # location of changed radius

        # 對陣列的值做累加
        csim = np.cumsum(i_sorted, dtype=float)
        # 累加的值
        tbin = csim[rind]
        # 算出累加的區間
        tbin[1:] -= csim[rind[:-1]]

        nr = rind - np.roll(rind, 1)
        nr = nr[1:]
        # 第一個值(圓心)不用除
        tbin[1:] /= nr

        return tbin

    def get_DXO_acutance(self):

        # read img
        I = self.dxo_roi_img.copy()
        # to gray level
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')

        # crop img to NxN square
        N = min(I.shape)

        # let N be the odd number
        if N % 2 == 0:
            N -= 1
        I = I[:N, :N]

        # compute I_hat(m, n)

        # Take the fourier transform of the image.
        I_hat = np.fft.fft2(I)

        # shift
        # [-N/2, N/2] => [0, N]
        # I(0,0) => I(N//2, N//2)
        I_hat = np.fft.fftshift(I_hat)

        # get the real part
        I_hat = np.abs(I_hat)

        # linear
        # I_hat = I_hat/np.mean(I)

        # I(0,0) => I(N//2, N//2) = N * N * E(I)
        # print(I_hat[N//2,N//2])
        # print(np.sum(I))
        # print(I_hat[N//2-1:N//2+2, N//2-1:N//2+2])

        # compute c(N)

        eta = -1.93
        Denominator = 0
        for m in range(0, N):
            for n in range(0, N):
                if m == N//2 and n == N//2:
                    continue
                Denominator += (1 / pow(((m-N//2)**2 + (n-N//2)**2), eta/2))
        cN = (I.var() / Denominator) * (N**4)

        # compute T_hat(m, n)

        T_hat = np.zeros((N, N))
        for m in range(0, N):
            for n in range(0, N):
                if m == N//2 and n == N//2:
                    continue
                T_hat[m, n] = cN / ((m-N//2)**2 + (n-N//2)**2)**(eta/2)
        # when m==0 and n == 0
        T_hat[N//2, N//2] = I_hat[N//2, N//2]

        # compute K(m, n)

        K = I_hat / T_hat
        # print(K[N//2, N//2])

        # compute MTF

        # The one-dimensional texture MTF is the average of over all directions.
        MTF = self.radialAverage(K)
        # print(MTF[:10])

        # compute CSF

        # contrast sensitivity function (CSF) can be used to weigh the
        # different spatial frequencies, leading to a single acutance value
        b = 0.2
        c = 0.8
        # CSF(v) = a * pow(v, c) * pow(e, -b*v)
        # ∫ CSF(v) dv = 1
        # ∫ a * pow(v, c) * pow(e, -b*v) dv = 1
        # a * ∫ pow(v, c) * pow(e, -b*v) dv = 1
        # a = 1 / ∫ pow(v, c) * pow(e, -b*v) dv
        a = 1 / np.sum([pow(v, c) * pow(e, -b*v) for v in range(MTF.shape[0])])
        CSF = [a * pow(v, c) * pow(e, -b*v) for v in range(MTF.shape[0])]

        # DXO book
        # a = 75
        # b = 0.2
        # c = 0.8
        # K = 34.05
        # CSF = [(a*pow(v, c) * e*pow(-b, v))/K for v in range(MTF.shape[0])]

        # compute Acutance
        A = np.sum([MTF[v] * CSF[v] for v in range(MTF.shape[0])])
        # print(A)

        # DXO book
        # A = np.sum([MTF[v] * CSF[v] for v in range(MTF.shape[0])])
        # A_r = np.sum([CSF[v] for v in range(MTF.shape[0])])
        # A = A/A_r

        return np.round(A, 4)

    # ###### colorcheck ######
    # def gen_colorcheck_coordinate(self):
    #     coor = []
    #     h, w, c = self.get_roi_img_by_roi_coordinate().shape
    #     square = self.square_rate*w
    #     padding = self.padding_rate*w

    #     start_h = self.start_h_rate*w
    #     for i in range(4):
    #         start_w = self.start_w_rate*w
    #         for j in range(6):
    #             coor.append([int(start_w), int(start_h), int(start_w+square), int(start_h+square)])
    #             start_w+=(square+padding)
    #         start_h+=(square+padding)

    #     return coor

    # def set_colorcheck_coordinate(self, coor):
    #     self.colorcheck_coordinate = coor

    # def get_colorcheck_roi_draw(self):
    #     color = (0, 0, 255) # red
    #     thickness = self.roi_img.shape[1]//200 # 寬度 (-1 表示填滿)

    #     img = self.roi_img.copy()
    #     for i in range(24):
    #         cv2.rectangle(img, (self.roi_coordinate[i][0], self.roi_coordinate[i][1]), (self.roi_coordinate[i][2], self.roi_coordinate[i][3]), color, thickness)
    #     return img
