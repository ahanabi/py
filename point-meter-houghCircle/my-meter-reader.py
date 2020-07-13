import os
import cv2
import numpy as np
from time import time

class METER(object):
    def __init__(self, path):
        self.path = path

    def readData(self):
        imgs_path = []
        for filename in os.listdir(self.path):
            if filename.endswith('.jpg'):
                filename = self.path + '/' + filename
                imgs_path.append(filename)
        return imgs_path

    def normalized_picture(self):
        img = cv2.imread(self.path)
        nor = cv2.resize(img, None, fx = 1, fy = 1, interpolation=cv2.INTER_LINEAR)
        y, x = nor.shape[:2]
        print("图片的长和宽为：", x, y)
        cv2.imshow('Normalized picture', nor)
        return nor

    def detect_circles(self, nor):
        gray = cv2.cvtColor(nor, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Graying pictures', gray)
        height, width = nor.shape[:2]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=80,
                                   minRadius=int(height * 0.05),
                                   maxRadius=int(height * 0.20))

        cir = nor.copy()
        cir_nor = nor.copy()
        a, b, c = circles.shape
        avg_x = 0
        avg_y = 0
        avg_r = 0
        for i in range(b):
            avg_x = avg_x + circles[0][i][0]
            avg_y = avg_y + circles[0][i][1]
            avg_r = avg_r + circles[0][i][2]
        avg_x = int(avg_x / (b))
        avg_y = int(avg_y / (b))
        avg_r = int(avg_r / (b))
        print("圆的半径 =", avg_r, " , 圆心坐标 =","(" ,avg_x, ",",avg_y, ")" )
        cv2.circle(cir, (avg_x, avg_y), avg_r, (0, 0, 255), 3, cv2.LINE_AA)   #圆环
        cv2.circle(cir, (avg_x, avg_y), 2, (255, 0, 0), 3, cv2.LINE_AA)   #圆心

        separation = 10.0
        interval = int(360 / separation)
        p1 = np.zeros((interval, 2))
        p2 = np.zeros((interval, 2))
        p_text = np.zeros((interval, 2))
        for i in range(0, interval):
            for j in range(0, 2):
                if (j % 2 == 0):
                    p1[i][j] = avg_x + 0.9 * avg_r * np.cos(separation * i * 3.14 / 180)  # point for lines
                else:
                    p1[i][j] = avg_y + 0.9 * avg_r * np.sin(separation * i * 3.14 / 180)
        text_offset_x = 10
        text_offset_y = 5
        for i in range(0, interval):
            for j in range(0, 2):
                if (j % 2 == 0):
                    p2[i][j] = avg_x + avg_r * np.cos(separation * i * 3.14 / 180)
                    p_text[i][j] = avg_x - text_offset_x + 1.2 * avg_r * np.cos((separation) * (
                            i + 9) * 3.14 / 180)
                else:
                    p2[i][j] = avg_y + avg_r * np.sin(separation * i * 3.14 / 180)
                    p_text[i][j] = avg_y + text_offset_y + 1.2 * avg_r * np.sin((separation) * (
                            i + 9) * 3.14 / 180)

        for i in range(0, interval):
            cv2.line(cir, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
            cv2.putText(cir, '%s' % (int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('circle', cir)
        #cir = cir[avg_y - avg_r:avg_y + avg_r, avg_x - avg_r:avg_x + avg_r]
        #cv2.imshow('circle_nor', cir)
        cir_nor = cir
        #cir_nor = cir_nor[avg_y - avg_r:avg_y + avg_r, avg_x - avg_r:avg_x + avg_r]
        y, x = cir_nor.shape[:2]
        y_s = int(368)
        x_s = x * y_s / y
        x_x = int(x_s)
        crop_size = (x_x, y_s)
        cir_nor = cv2.resize(cir_nor, crop_size, interpolation=cv2.INTER_LINEAR)
        print("调整后图片长和宽为:",500, 375)
        y_y = int(y_s/2)
        print("调整后圆的半径 =", 184, " , 圆心坐标 =", "(", 271, ",", 185, ")")
        return (cir_nor)


    def color_conversion(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Graying_nor', gray)
        return gray

    # 中值滤波去噪
    def median_filter(self,img):
        median = cv2.medianBlur(img, 1)  # 中值滤波
        cv2.imshow('Median filter', median)
        return median

    # 双边滤波去噪
    def bilateral_filter(self,img):
        bilateral = cv2.bilateralFilter(img, 9, 50, 50)
        cv2.imshow('Bilateral filter', bilateral)
        return bilateral

    # 高斯滤波去噪
    def gaussian_filter(self,img):
        gaussian = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imshow('Gaussian filter', gaussian)
        return gaussian
    '''
    # 图像二值化
    def binary_image(self,img):
        # 应用5种不同的阈值方法
        # ret, th1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        # ret, th2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        # ret, th3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        # ret, th4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
        # ret, th5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
        # titles = ['Gray', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        # images = [img_gray, th1, th2, th3, th4, th5]
        # 使用Matplotlib显示
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(images[i], 'gray')
        #     plt.title(titles[i], fontsize=8)
        #     plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
        # plt.show()

        # Otsu阈值
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        cv2.imshow('Binary image', th)
        return th
        '''
    # 边缘检测
    def candy_image(self,img):
        edges = cv2.Canny(img, 60, 143, apertureSize=3)
        cv2.imshow('canny', edges)
        return edges
    '''
    # 开运算：先腐蚀后膨胀
    def open_operation(self,img):
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # 椭圆结构
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))     # 十字形结构
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        cv2.imshow('Open operation', opening)
        return opening
    '''

    def detect_pointer(self, cir_nor, candy):
        '''
        height, width = gray.shape[:2]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50,
                                   minRadius=int(height * 0.20),
                                   maxRadius=int(height * 0.50))
        a, b, c = circles.shape
        avg_x = 0
        avg_y = 0
        avg_r = 0
        for i in range(b):
            avg_x = avg_x + circles[0][i][0]
            avg_y = avg_y + circles[0][i][1]
            avg_r = avg_r + circles[0][i][2]
        avg_x = int(avg_x / (b))
        avg_y = int(avg_y / (b))
        avg_r = int(avg_r / (b))
        '''
        y, x = cir_nor.shape[:2]
        rho = 1
        theta = np.pi / 180
        minLineLength = y * 0.4
        max_line_gap = y * 0.05
        threshold = 66
        lines = cv2.HoughLinesP(candy, rho, theta, threshold, minLineLength=minLineLength,maxLineGap=max_line_gap)
        #lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(cir_nor, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('lines', cir_nor)

        '''
        final_line_list = []
        diff1LowerBound = 0.1
        diff1UpperBound = 0.5
        diff2LowerBound = 0.8
        diff2UpperBound = 1.0
        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                diff1 = np.sqrt((x1 - avg_x) ** 2 + (y1 - avg_y) ** 2)
                diff2 = np.sqrt((x2 - avg_x) ** 2 + (y2 - avg_y) ** 2)

                if (diff1 > diff2).any():
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp

                if (((diff1<diff1UpperBound*avg_r) and (diff1>diff1LowerBound*avg_r) and (diff2<diff2UpperBound*avg_r)) and (diff2>diff2LowerBound*avg_r)).all():
                    #print('diff1UpperBound*r=', diff1UpperBound * avg_r)
                    #print('diff1LowerBound*r=', diff1LowerBound * avg_r)
                    #print('diff2UpperBound*r=', diff2UpperBound * avg_r)
                    #print('diff2LowerBound*r=', diff2LowerBound * avg_r)
                    #print('diff1=', diff1)
                    #print('diff2=', diff2)
                    #line_length = dist_2_pts(x1, y1, x2, y2)
                    #print('直线长度为：', line_length)
                    #print('直线坐标为：', [x1, y1, x2, y2])
                    final_line_list.append([x1, y1, x2, y2])

            for i in range(0,len(final_line_list)):
                x1 = final_line_list[i][0]
                y1 = final_line_list[i][1]
                x2 = final_line_list[i][2]
                y2 = final_line_list[i][3]
                cv2.line(cir, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('lines', cir)

        dist_pt_0 = np.sqrt((x1 - avg_x) ** 2 + (y1 - avg_y) ** 2)
        dist_pt_1 = np.sqrt((x2 - avg_x) ** 2 + (y2 - avg_y) ** 2)
        if (dist_pt_0 >= dist_pt_1):
            x_angle = x1 - avg_x
            y_angle = avg_y - y1
        else:
            x_angle = x2 - avg_x
            y_angle = avg_y - y2

        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        res = np.rad2deg(res)
        if x_angle > 0 and y_angle > 0:
            final_angle = 270 - res
        if x_angle < 0 and y_angle > 0:
            final_angle = 90 - res
        if x_angle < 0 and y_angle < 0:
            final_angle = 90 - res
        if x_angle > 0 and y_angle < 0:
            final_angle = 270 - res

        print('指针角度为:',final_angle)
        '''

    def iden_pic(self):
        d = 21.3144136564651325
        t = 0.42546125102102562
        t1 = time()
        image = METER(self.path)
        nor = image.normalized_picture()
        t1 = time()
        cir_nor = image.detect_circles(nor)
        t2 = time()
        d = t2 - t1
        print("Hough圆检测的时间为", d)
        gray = image.color_conversion(cir_nor)
        #binary = image.binary_image(gray)
        median = image.median_filter(gray)
        bilateral = image.bilateral_filter(median)
        gaussian = image.gaussian_filter(gray)
        candy = image.candy_image(gaussian)
        #open = image.open_operation(median)
        pointer = image.detect_pointer(cir_nor, candy)
        print('刻度为:' , d, '℃')
        print('程序运行时间为:', t)
        t2 = time()
        t = t2 - t1
        cv2.waitKey(0)
if __name__ == '__main__':
    input_path = input('输入图片：')
    METER(input_path).iden_pic()