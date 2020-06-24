import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import random
import json
import argparse

sys.path.append(os.getcwd())
import ai_api.utils.image_helpler as image_helpler
import ai_api.utils.file_helper as file_helper
from ai_api.gauge.gauge_model import GaugeModel

# 把模型的变量分布在哪个GPU上给打印出来
# tf.debugging.set_log_device_placement(True)

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='image_data')
parser.add_argument('--batch_size', default=6, type=int)
args = parser.parse_args()

model = GaugeModel.GetStaticModel()


def generator(file_list, batch_size):
    X = []
    Y = []
    # 用于数据平均
    value_index = 0
    skip_index = 0
    while True:
        # 打乱列表顺序
        random_list = np.array(file_list)
        np.random.shuffle(random_list)
        for file_path in random_list:
            try:
                # 读取json文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                # json文件目录
                json_dir = os.path.dirname(file_path)
                # json文件名
                json_name = os.path.basename(file_path)
                # 图片路径
                image_path = os.path.join(
                    json_dir, json_data['imagePath'].replace('\\', '/'))
                # 图片文件名
                image_name = os.path.basename(image_path)
                # 值
                value = json_data['shapes'][0]["label"].split('_')
                if len(value)>1:
                    value = float(value[1])
                else:
                    value = float(value[0])
                if skip_index > 50:
                    skip_index = 0
                    # 值下标加1
                    value_index += 1
                    if value_index > 10:
                        value_index = 0
                # 值平均
                if (value_index-1)*10 >= value or value > value_index*10:
                    skip_index += 1
                    continue
                # print('添加：', (value // 10), value_index)
                # print('添加：', value)
                skip_index = 0
                # 值下标加1
                value_index += 1
                if value_index > 10:
                    value_index = 0
                # 原始点列表
                json_points = np.float32(json_data['shapes'][0]['points'])
                # 点匹配
                point_center_x = (min(json_points[:, 0]) + max(json_points[:, 0])) / 2
                point_center_y = (min(json_points[:, 1]) + max(json_points[:, 1])) / 2
                for p in json_points:
                    if p[0] < point_center_x and p[1] < point_center_y:
                        pointLT = p
                    elif p[0] > point_center_x and p[1] < point_center_y:
                        pointRT = p
                    elif p[0] < point_center_x and p[1] > point_center_y:
                        pointLB = p
                    elif p[0] > point_center_x and p[1] > point_center_y:
                        pointRB = p
                points = np.float32([pointLT, pointLB, pointRT, pointRB])
                # print('points:', points)
                # 读取图片
                img = image_helpler.fileToOpencvImage(image_path)
                # 缩放图片
                img, points, _ = image_helpler.opencvProportionalResize(
                    img, (400, 400), points=points)
                # print('imgType:', type(img))
                # width, height = image_helpler.opencvGetImageSize(img)
                # print('imgSize:', width, height)
                # 获取随机变换图片及标签
                random_img, target_data = model.get_random_data(
                    img, value, target_points=points)
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
            except Exception as expression:
                print('异常：', expression, file_path)


def train():
    '''训练'''
    train_path = args.file_path
    file_list = file_helper.ReadFileList(train_path, r'.json$')
    print('图片数：', len(file_list))
    # 训练参数
    batch_size = args.batch_size
    steps_per_epoch = 200
    epochs = 500
    model.fit_generator(generator(file_list, batch_size),
                        steps_per_epoch, epochs, auto_save=True)


def main():
    train()


if __name__ == '__main__':
    main()
