from django.http import HttpResponse
import json
import base64
import io
import cv2
import random
import numpy as np
import math
import time
from ai_api.gauge.gauge_model import GaugeModel
import ai_api.utils.image_helpler as image_helpler

model = GaugeModel.GetStaticModel()
# 训练次数，用于保存模型
train_num = 0
save_num = time.time()


def gauge_train(request):
    '''训练模型'''
    global train_num
    request_data = json.loads(request.body)
    # print('request_data:', request_data)
    img_data = request_data['img_data'].split(',')[1]
    img_data = image_helpler.base64ToBytes(img_data)
    img = image_helpler.bytesToOpencvImage(img_data)
    value = request_data['value']
    img_name = str(value)+'.jpg'
    path = "./image_data/" + img_name
    # with open(path, 'wb') as f:
    #     f.write(img_data)

    # print('imgType:', type(img))
    # width, height = image_helpler.opencvGetImageSize(img)
    # print('imgSize:', width, height)
    # 获取随机变换图片及标签
    random_img, target_data = model.get_random_data(img, value)
    # 增加一个维度
    random_img = np.expand_dims(random_img, 0)
    target_data = np.expand_dims(target_data, 0)
    print('random_img:', random_img.shape, np.max(random_img))
    print('target_data:', target_data.shape, np.max(target_data))
    print('value:', value)
    is_train = True
    max_train = 0
    while is_train:
        output_value = model.predict(random_img)
        print('output_value:', output_value)
        print('target_data:', target_data)
        if abs(output_value[0, 0]-value) > 0.02:
            print('训练')
            loss = model.train_step(random_img, target_data)
            print('loss:', loss)
            # is_train = (random.random() > 0.1)
            is_train = False
            if max_train > 50:
                break
            train_num = train_num + 1
            max_train = max_train + 1
            # if train_num % 100 == 0:
            #     model.save_model()
        else:
            print('跳过训练')
            is_train = False
    jsonObj = {
        "value": output_value.numpy().tolist(),
    }
    return HttpResponse(json.dumps(jsonObj), content_type="application/json")


def gauge_predict(request):
    '''识别'''
    global train_num
    request_data = json.loads(request.body)
    # print('request_data:', request_data)
    read = request_data['read']
    img_data = request_data['img_data'].split(',')[1]
    img_data = image_helpler.base64ToBytes(img_data)
    img = image_helpler.bytesToOpencvImage(img_data)
    # 缩放图片
    img, _, _ = image_helpler.opencvProportionalResize(img, (400, 400))

    # print('imgType:', type(img))
    # width, height = image_helpler.opencvGetImageSize(img)
    # print('imgSize:', width, height)
    # 获取随机变换图片及标签
    random_img = img
    if read != 1:
        print('随机变换')
        random_img = model.get_random_image(random_img)
    # 最后输出图片
    predict_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2RGB)
    # 调整参数范围
    predict_img = predict_img.astype(np.float32)
    predict_img = predict_img / 255
    # 增加一个维度
    predict_img = np.expand_dims(predict_img, 0)
    output_value = model.predict(predict_img)
    print('output_value:', output_value)
    # 透视变换
    org = np.float32([[output_value[0, 1]*400, output_value[0, 2]*400],
                      [output_value[0, 3]*400, output_value[0, 4]*400],
                      [output_value[0, 5]*400, output_value[0, 6]*400],
                      [output_value[0, 7]*400, output_value[0, 8]*400]])
    dst = np.float32([[50, 50],
                      [50, 350],
                      [350, 50],
                      [350, 350]])
    org = dst + org
    perspective_img = image_helpler.opencvPerspectiveP(random_img, org, dst)
    jsonObj = {
        "value": output_value.numpy().tolist(),
        'random_img': image_helpler.bytesTobase64(image_helpler.opencvImageToBytes(random_img)),
        'perspective_img': image_helpler.bytesTobase64(image_helpler.opencvImageToBytes(perspective_img)),
    }
    # print('jsonObj:',jsonObj)
    return HttpResponse(json.dumps(jsonObj), content_type="application/json")


def gauge_save(request):
    '''保存训练图片'''
    global save_num
    save_num += 0.000001
    request_data = json.loads(request.body)
    # print('request_data:', request_data)
    img_data = request_data['img_data'].split(',')[1]
    img_data = image_helpler.base64ToBytes(img_data)
    img = image_helpler.bytesToOpencvImage(img_data)
    value = request_data['value']
    img_name = ('%s_%.2f.jpg') % (str(save_num), value)
    path = "./image_data/" + img_name
    # with open(path, 'wb') as f:
    #     f.write(img_data)

    image_helpler.opencvImageToFile(path, img)

    jsonObj = {
        "value": value,
    }
    return HttpResponse(json.dumps(jsonObj), content_type="application/json")
