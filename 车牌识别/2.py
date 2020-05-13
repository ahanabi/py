def preprocessing(img):
    '''
    预处理函数
    '''
    m=400 * img.shape[0] / img.shape[1]

    #压缩图像
    img=cv2.resize(img,(400,int(m)),interpolation=cv2.INTER_CUBIC)

    #BGR转换为灰度图像
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print('gray_img.shape',gray_img.shape)

    #灰度拉伸
    #如果一幅图像的灰度集中在较暗的区域而导致图像偏暗，可以用灰度拉伸功能来拉伸(斜率>1)物体灰度区间以改善图像；
    # 同样如果图像灰度集中在较亮的区域而导致图像偏亮，也可以用灰度拉伸功能来压缩(斜率<1)物体灰度区间以改善图像质量
    stretchedimg=stretching(gray_img)#进行灰度拉伸，是因为可以改善图像的质量
    print('stretchedimg.shape',stretchedimg.shape)

    '''进行开运算，用来去除噪声'''
    r=15
    h=w=r*2+1
    kernel=np.zeros((h,w),np.uint8)
    cv2.circle(kernel,(r,r),r,1,-1)
    #开运算
    openingimg=cv2.morphologyEx(stretchedimg,cv2.MORPH_OPEN,kernel)
    #获取差分图，两幅图像做差  cv2.absdiff('图像1','图像2')
    strtimg=cv2.absdiff(stretchedimg,openingimg)
    cv2.imshow("stretchedimg",stretchedimg)
    cv2.imshow("openingimg1",openingimg)
    cv2.imshow("strtimg",strtimg)
    cv2.waitKey(0)

    #图像二值化
    binaryimg=allbinaryzation(strtimg)
    cv2.imshow("binaryimg",binaryimg)
    cv2.waitKey(0)

    #canny边缘检测
    canny=cv2.Canny(binaryimg,binaryimg.shape[0],binaryimg.shape[1])
    cv2.imshow("canny",canny)
    cv2.waitKey(0)

    '''保留车牌区域，消除其他区域，从而定位车牌'''
    #进行闭运算
    kernel=np.ones((5,23),np.uint8)
    closingimg=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("closingimg",closingimg)

    #进行开运算
    openingimg=cv2.morphologyEx(closingimg,cv2.MORPH_OPEN,kernel)
    cv2.imshow("openingimg2",openingimg)

    #再次进行开运算
    kernel=np.ones((11,6),np.uint8)
    openingimg=cv2.morphologyEx(openingimg,cv2.MORPH_OPEN,kernel)
    cv2.imshow("openingimg3",openingimg)
    cv2.waitKey(0)

    #消除小区域，定位车牌位置
    rect=locate_license(openingimg,img)#rect包括轮廓的左上点和右下点，长宽比以及面积

    return rect,img