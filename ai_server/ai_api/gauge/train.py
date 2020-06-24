import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import random

sys.path.append(os.getcwd())
from ai_api.gauge.gauge_model import gauge_model
import ai_api.utils.file_helper as file_helper
import ai_api.utils.image_helpler as image_helpler

# 把模型的变量分布在哪个GPU上给打印出来
# tf.debugging.set_log_device_placement(True)

model = gauge_model()


def generator(file_list, batch_size):
    X = []
    Y = []
    while True:
        # 打乱列表顺序
        random_list = np.array(file_list)
        np.random.shuffle(random_list)
        for file_path in random_list:
            img = image_helpler.fileToOpencvImage(file_path)
            # 缩放图片
            img, _, _ = image_helpler.opencvProportionalResize(img, (400, 400))
            value = os.path.basename(file_path).split('_')[1]
            value = float(value[:-4])
            # print('imgType:', type(img))
            # width, height = image_helpler.opencvGetImageSize(img)
            # print('imgSize:', width, height)
            # 获取随机变换图片及标签
            random_img, target_data = model.get_random_data(img, value)
            X.append(random_img)
            # Y.append([value])
            Y.append(target_data)
            if len(Y) == batch_size:
                result_x = np.array(X)
                result_y = np.array(Y)
                # print('generator', result_x.shape, result_y.shape)
                yield result_x, result_y
                X = []
                Y = []


def train():
    '''训练'''
    file_list = file_helper.ReadFileList('image_data(虚拟表原图)', r'.jpg$')
    print('图片数：', len(file_list))
    # 训练参数
    batch_size = 6
    steps_per_epoch = 100
    epochs = 500
    model.fit_generator(generator(file_list, batch_size),
                        steps_per_epoch, epochs, auto_save=True)


def main():
    train()

if __name__ == '__main__':
    main()
