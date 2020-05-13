import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import cos, pi, sin
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
        y, x = img.shape[:2]
        y_s = 1200
        x_s = x * y_s / y
        x_x = int(x_s)
        crop_size = (x_x, y_s)
        nor = cv2.resize(img, crop_size, interpolation=cv2.INTER_LINEAR)
        #nor = cv2.resize(img, None, fx = 1, fy = 1, interpolation=cv2.INTER_LINEAR)
        y, x = nor.shape[:2]
        print ("图片的长和宽为：", x, y)
        cv2.imshow('Normalized picture', nor)
        return nor

    def color_conversion(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Graying pictures', gray)
        return gray

    def detect_circles(self, gray, img):
        height, width = img.shape[:2]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50,
                                   minRadius=int(height * 0.25),
                                   maxRadius=int(height * 0.50))

        cir = img.copy()
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

        cv2.circle(cir, (avg_x, avg_y), avg_r, (0, 0, 255), 3, cv2.LINE_AA)   #圆环
        cv2.circle(cir, (avg_x, avg_y), 2, (255, 0, 0), 3, cv2.LINE_AA)   #圆心

        print("圆半径及圆心坐标为:", avg_r, avg_x)
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

        cir = cir[avg_y - avg_r:avg_y + avg_r, avg_x - avg_r:avg_x + avg_r]
        cv2.imshow('circle', cir)
        return (cir)

    def v2_by_k_means(self,cir):
        img = cir.copy()
        original_img = np.array(img, dtype=np.float64)
        src = cir.copy()
        delta_y = int(original_img.shape[0] * (0.4))
        delta_x = int(original_img.shape[1] * (0.4))
        original_img = original_img[delta_y:-delta_y, delta_x:-delta_x]
        h, w, d = src.shape
        # print(w, h, d)
        dts = min([w, h])
        # print(dts)
        r2 = (dts / 2) ** 2
        c_x, c_y = w / 2, h / 2
        a: np.ndarray = original_img[:, :, 0:3].astype(np.uint8)
        # 获取尺寸(宽度、长度、深度)
        height, width = original_img.shape[0], original_img.shape[1]
        depth = 3
        # print(depth)
        image_flattened = np.reshape(original_img, (width * height, depth))
        image_array_sample = shuffle(image_flattened, random_state=0)
        estimator = KMeans(n_clusters=2, random_state=0)
        estimator.fit(image_array_sample)
        src_shape = src.shape
        new_img_flattened = np.reshape(src, (src_shape[0] * src_shape[1], depth))
        cluster_assignments = estimator.predict(new_img_flattened)
        compressed_palette = estimator.cluster_centers_
        # print(compressed_palette)
        a = np.apply_along_axis(func1d=lambda x: np.uint8(compressed_palette[x]), arr=cluster_assignments, axis=0)
        img = a.reshape(src_shape[0], src_shape[1], depth)
        # print(compressed_palette[0, 0])
        threshold = (compressed_palette[0, 0] + compressed_palette[1, 0]) / 2
        img[img[:, :, 0] > threshold] = 255
        img[img[:, :, 0] < threshold] = 0
        # cv2.imshow('sd0', img)
        for x in range(w):
            for y in range(h):
                distance = ((x - c_x) ** 2 + (y - c_y) ** 2)
                if distance > r2:
                    pass
                    img[y, x] = (255, 255, 255)
        cv2.imshow('kmeans', img)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        return img

    def get_pointer_rad(self, img):
        '''获取角度'''
        shape = img.shape
        c_y, c_x, depth = int(shape[0] / 2), int(shape[1] / 2), shape[2]
        x1 = c_x + c_x * 0.8
        src = img.copy()
        freq_list = []
        for i in range(3601):
            x = (x1 - c_x) * cos(i * pi / 1800) + c_x
            y = (x1 - c_x) * sin(i * pi / 1800) + c_y
            temp = src.copy()
            cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=3)
            t1 = img.copy()
            t1[temp[:, :, 2] == 255] = 255
            c = img[temp[:, :, 2] == 255]
            points = c[c == 0]
            i = i / 10
            freq_list.append((len(points), i))
            #cv2.imshow('d0', temp)
            cv2.imshow('zhixian', t1)
            cv2.waitKey(1)
        print('最大重合数量和对应角度:', max(freq_list, key=lambda x: x[0]))
        #cv2.destroyAllWindows()
        return max(freq_list, key=lambda x: x[0])

    def iden_pic(self):
        t1 = time()
        image = METER(self.path)
        nor = image.normalized_picture()
        gray = image.color_conversion(nor)
        cir = image.detect_circles(gray, nor)
        img = image.v2_by_k_means(cir)
        rad = image.get_pointer_rad(img)
        t2 = time()
        t = t2 - t1
        print('程序运行时间为:', t)
        cv2.waitKey(0)
if __name__ == '__main__':
    input_path = input('输入图片：')
    METER(input_path).iden_pic()