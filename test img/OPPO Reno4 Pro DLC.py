import sys
sys.path.append("..")
from myPackage.ROI import *

def white_balance(img):
    b, g, r = cv2.split(img)
    """
    YUV空間
    """
    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    # print(max_y)
    for i in range(m):
        for j in range(n):
            sum_u = sum_u + u[i][j]
            sum_v = sum_v + v[i][j]

    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
    for i in range(m):
        for j in range(n):
            du = du + np.abs(u[i][j] - avl_u)
            dv = dv + np.abs(v[i][j] - avl_v)

    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5  # 如果該值過大過小，色溫向兩極端發展
    for i in range(m):
        for j in range(n):
            value = 0
            if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
                    v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
                value = 1
            else:
                value = 0

            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
            ysum += 1
    # print(yhistogram.shape)
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.1 * ysum:    # 取前10%的亮點爲計算值，如果該值過大易過曝光，該值過小調整幅度小
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r = sum_r + r[i][j]
                sum_g = sum_g + g[i][j]
                sum_b = sum_b + b[i][j]
                num_rgb += 1

    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb

    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            if b_point>255:
                b[i][j] = 255
            else:
                if b_point<0:
                    b[i][j] = 0
                else:
                    b[i][j] = b_point
            if g_point>255:
                g[i][j] = 255
            else:
                if g_point<0:
                    g[i][j] = 0
                else:
                    g[i][j] = g_point
            if r_point>255:
                r[i][j] = 255
            else:
                if r_point<0:
                    r[i][j] = 0
                else:
                    r[i][j] = r_point

    return cv2.merge([b, g, r])

roi = ROI()


roi.img = cv2.imread("OPPO Find X2 DLC/A_5_1.246.jpg")
roi.set_dxo_roi_img(TEST=False)
# roi.roi_img = white_balance(roi.roi_img)
# roi.set_gamma()
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/A_5_1.237.jpg")
roi.set_dxo_roi_img(TEST=False)
# roi.roi_img = white_balance(roi.roi_img)
# roi.set_gamma()
print(roi.get_Imatest_any_sharp(roi.roi_img))


roi.img = cv2.imread("OPPO Find X2 DLC/A_20_1.290.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/A_20_1.241.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))


roi.img = cv2.imread("OPPO Find X2 DLC/A_100_1.288.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/A_100_1.510.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Find X2 DLC/A_300_1.435.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/A_300_1.605.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Find X2 DLC/D65_1000_1.531.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/D65_1000_1.773.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Find X2 DLC/H_1_1.090.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/H_1_1.320.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Find X2 DLC/TL84_20_1.414.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/TL84_20_1.363.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Find X2 DLC/TL84_100_1.450.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/TL84_100_1.546.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Find X2 DLC/TL84_300_1.483.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/TL84_300_1.613.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))

roi.img = cv2.imread("OPPO Reno4 Pro DLC/TL84_1000_1.751.jpg")
roi.set_dxo_roi_img(TEST=False)
print(roi.get_Imatest_any_sharp(roi.roi_img))
