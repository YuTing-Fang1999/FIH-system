from tkinter.messagebox import NO
import cv2
import numpy as np
from scipy.signal import convolve2d
import math
from math import e

class ROI:
    # 建構式
    def __init__(self):
        self.img = None
        self.roi_img = None
        self.roi_coordinate = None

    def set_roi_img(self, img, roi_coordinate):
        self.img = img
        self.roi_coordinate = roi_coordinate
        coor = self.roi_coordinate
        self.roi_img = self.img[int(coor.r1):int(coor.r2), int(coor.c1):int(coor.c2), :]
        

    ###### sharpness ######
    def get_sharpness(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        return np.round(np.sqrt(cv2.Laplacian(I, cv2.CV_64F).var()), 4)
    
    def get_noise(self):  
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        H, W = I.shape

        M = [[1, -2, 1],
           [-2, 4, -2],
           [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
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
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gamma=0.5
        invGamma = 1.0 / gamma
        I = np.array(((I / 255.0) ** invGamma) * 255) # linearized

        sobelx = cv2.Sobel(I, cv2.CV_16S, 1, 0)  # x方向梯度 ksize默認為3x3
        sobely = cv2.Sobel(I, cv2.CV_16S, 0, 1)  # y方向梯度

        abx = np.abs(sobelx)  # 取sobelx絕對值
        aby = np.abs(sobely)  # 取sobely絕對值

        sx = np.mean(abx)/np.mean(I)
        sy = np.mean(aby)/np.mean(I)
        
        sTotal = np.sqrt(sx*sx + sy*sy)

        return np.round(sTotal, 4)

    def get_gamma_Sobel(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gamma=0.5
        invGamma = 1.0 / gamma
        I = np.array(((I / 255.0) ** invGamma) * 255) # linearized

        sobelx = cv2.Sobel(I, cv2.CV_16S, 1, 0)  # x方向梯度 ksize默認為3x3
        sobely = cv2.Sobel(I, cv2.CV_16S, 0, 1)  # y方向梯度

        var = np.mean(sobelx**2 + sobely**2)

        return np.round(np.sqrt(var), 4)

    def get_gamma_Laplacian(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gamma=0.5
        invGamma = 1.0 / gamma
        I = np.array(((I / 255.0) ** invGamma) * 255) # linearized

        return np.round(np.sqrt(cv2.Laplacian(I, cv2.CV_64F, ksize=1).var()), 4)

    def get_H(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gamma=0.5
        invGamma = 1.0 / gamma
        I = np.array(((I / 255.0) ** invGamma) * 255) # linearized

        sobelx = cv2.Sobel(I, cv2.CV_16S, 1, 0, ksize=1)  # x方向梯度 ksize默認為3x3

        return np.round(np.sqrt(sobelx.var()), 4)

    def get_V(self):
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
        gamma=0.5
        invGamma = 1.0 / gamma
        I = np.array(((I / 255.0) ** invGamma) * 255) # linearized

        sobely = cv2.Sobel(I, cv2.CV_16S, 0, 1, ksize=1)  # y方向梯度
        # sobely = cv2.Sobel(sobely, cv2.CV_16S, 0, 1)  # y方向梯度

        return np.round(np.sqrt(sobely.var()), 4)

    
    ############ DXO deadleaves #############

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
        rind = np.where(deltar!=0)[0]       # location of changed radius

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

    
    def get_DXO_accurate(self):

        # read img
        I = self.roi_img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')

        # crop img to NxN square
        N = min(I.shape)

        # let N be the odd number
        if N%2==0: N-=1
        I = I[:N, :N] 

        ############################ compute I_hat(m, n) 

        # Take the fourier transform of the image.
        I_hat = np.fft.fft2(I)

        # shift
        # [-N/2, N/2] => [0, N]
        # I(0,0) => I(N//2, N//2)
        I_hat = np.fft.fftshift(I_hat) 

        # get the real part
        I_hat = np.abs(I_hat)

        # I(0,0) => I(N//2, N//2) = N * N * E(I)
        # print(I_hat[N//2,N//2])
        # print(np.sum(I))

        ############################ compute c(N)

        eta = -1.93
        Denominator = 0
        for m in range(0, N):
            for n in range(0, N):
                if m==N//2 and n == N//2: continue
                Denominator += (1 / pow(((m-N//2)**2 + (n-N//2)**2), eta/2))
        cN = (I.var() / Denominator) * (N**4)

        ############################ compute T_hat(m, n)

        T_hat = np.zeros((N,N))
        for m in range(0, N):
            for n in range(0, N):
                if m==N//2 and n == N//2: continue
                T_hat[m,n] = cN / ((m-N//2)**2 + (n-N//2)**2)**(eta/2)
        # when m==0 and n == 0
        T_hat[N//2,N//2] = I_hat[N//2,N//2]

        ############################ compute K(m, n)

        K = I_hat / T_hat 
        # print(K[N//2, N//2])

        ############################ compute MTF

        # The one-dimensional texture MTF is the average of over all directions.
        MTF = self.radialAverage(K)

        ############################ compute CSF

        # contrast sensitivity function (CSF) can be used to weigh the
        # different spatial frequencies, leading to a single acutance value
        b = 0.2
        c = 0.8

        # CSF(v) = a * pow(v, c) * pow(e, -b*v)
        # ∫ CSF(v) dv = 1 
        # ∫ a * pow(v, c) * pow(e, -b*v) dv = 1
        # a * ∫ pow(v, c) * pow(e, -b*v) dv = 1
        # a = 1 / ∫ pow(v, c) * pow(e, -b*v) dv
        a = 1 / np.sum([ pow(v, c) * pow(e, -b*v) for v in range(MTF.shape[0])])
        CSF = [ a * pow(v, c) * pow(e, -b*v) for v in range(MTF.shape[0])]

        ############################ compute Acutance

        A = np.sum([ MTF[v] * CSF[v] for v in range(MTF.shape[0])])
        # print(A)

        return A

    
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

    
