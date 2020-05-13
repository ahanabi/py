import numpy as np
import os
import cv2

# 设置图片路径，该路径下包含了14张jpg格式的照片，名字依次为0.jpg, 1.jpg, 2.jpg,...,14.jpg
DATADIR="D:\ABCD\ToolBox"
path=os.path.join(DATADIR)
#使用os.listdir(path)函数，返回path路径下所有文件的名字，以及文件夹的名字，
#例如，执行下行代码后，img_list是一个list，值为['0.jpg','1.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg',
#'2.jpg','3.jpg','4.jg', '5.jpg', '6.jpg', '7.jpg',
#'8.jpg', '9.jpg']，注意这个顺序并没有按照从小到大的顺序排列'''
img_list=os.listdir(path)
ind=0
for i in img_list:
	#调用cv2.imread读入图片，读入格式为IMREAD_COLOR'''
    img_array=cv2.imread(os.path.join(path,i), cv2.IMREAD_COLOR)
    #'''调用cv2.resize函数resize图片'''
    new_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_name=str(ind)+'.jpg'
    #'''生成图片存储的目标路径'''
    save_path='D:\\ABCD\\resized\\'+str(ind)+'.jpg'
    ind=ind+1
    #'''调用cv.2的imwrite函数保存图片'''
    cv2.imwrite(save_path,new_array)


