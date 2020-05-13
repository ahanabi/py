def carimg_make(img):
    # 预处理图像
    rect, afterimg = preprocessing(img)  # 其实包括了对车牌定位
    print("rect:", rect)

    # 框出车牌
    cv2.rectangle(afterimg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('afterimg1', afterimg)

    # 分割车牌与背景
    cutimg = cut_license(afterimg, rect)
    cv2.imshow('cutimg', cutimg)

    # 二值化生成黑白图
    thresh = lice_binarization(cutimg)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    # 分割字符
    '''
    判断底色和字色
    '''
    # 记录黑白像素总和
    white = []  # 记录每一列的白色像素总和
    black = []  # 记录每一列的黑色像素总和
    height = thresh.shape[0]  # 263
    width = thresh.shape[1]  # 400
    white_max = 0  # 仅保存每列，取列中白色最多的像素总数
    black_max = 0  # 仅保存每列，取列中黑色最多的像素总数
    # 计算每一列的黑白像素总和
    for i in range(width):
        line_white = 0  # 这一列白色总数
        line_black = 0  # 这一列黑色总数
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)
        print('white_max', white_max)
        print('black_max', black_max)
    # arg为true表示黑底白字，False为白底黑字
    arg = True
    if black_max < white_max:
        arg = False
    # 分割车牌的数字
    n  = 1
    start = 1
    end = 2
    s_width = 28
    s_height = 28
    temp = 1
    while n < width - 2:
        n += 1
        # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):#这点没有理解透彻
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            print("start" + str(start))
            print("end" + str(end))
            # 思路就是从左开始检测匹配字符，若宽度（end - start）小与20则认为是左侧白条 pass掉  继续向右识别，否则说明是
            # 省份简称，剪切，压缩 保存，还有一个当后五位有数字 1 时，他的宽度也是很窄的，所以就直接认为是数字 1 不需要再
            # 做预测了（不然很窄的 1 截切  压缩后宽度是被拉伸的），shutil.copy()函数是当检测
            # 到这个所谓的 1 时，从样本库中拷贝一张 1 的图片给当前temp下标下的字符
            # if end - start > 5:
            #     print("end - start" + str(end - start))
            if end - start > 5:
                cj = thresh[1:height, start:end]
                print("result/%s.jpg" % (n))
                cv2.imwrite('img/{0}.bmp'.format(n), cj)
                #对分割出的数字、字母进行裁剪
                b_img = cv2.resize(cj, None, fx=5, fy=3)
                contours, hierarchy = cv2.findContours(b_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                block = []
                for c in contours:
                    # 找出轮廓的左上点和右下点，由此计算它的面积和长度比
                    r = find_rectangle(c)  # 里面是轮廓的左上点和右下点
                    a = (r[2] - r[0]) * (r[3] - r[1])  # 面积
                    s = (r[2] - r[0]) / (r[3] - r[1])  # 长度比
                    block.append([c, r, a, s])
                block1 = sorted(block, key=lambda block: block[2])[-1:]
                # rect = cv2.minAreaRect(block2)
                # box1 = np.int0(cv2.boxPoints(rect))
                box = block1[0][1]
                y_mia = box[0]  # y_mia
                x_min = box[1]  # x_min
                y_max = box[2]  # y_max
                x_max = box[3]  # x_max
                cropImg = b_img[x_min:x_max, y_mia:y_max]  # crop the image
                cv2.imwrite('img_test/{0}.bmp'.format(n), cropImg)
                cv2.imshow('cutlicense', cj)
                cv2.imshow("charecter",cropImg)
                cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

def locate_license(img,afterimg):
    '''
    定位车牌号
    '''
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_copy = afterimg.copy()
    img_cont = cv2.drawContours(img_copy,contours,-1,(255,0,0),6)
    cv2.imshow("img_cont",img_cont)
    cv2.waitKey(0)
    #找出最大的三个区域
    block=[]
    for c in contours:
        #找出轮廓的左上点和右下点，由此计算它的面积和长度比
        r=find_rectangle(c)#里面是轮廓的左上点和右下点
        a=(r[2]-r[0])*(r[3]-r[1])   #面积
        s=(r[2]-r[0])/(r[3]-r[1])   #长度比

        block.append([r,a,s])
    #选出面积最大的3个区域
    block=sorted(block,key=lambda b: b[1])[-3:]

    #使用颜色识别判断找出最像车牌的区域
    maxweight,maxindex=0,-1
    for i in range(len(block)):#len(block)=3
        b=afterimg[block[i][0][1]:block[i][0][3],block[i][0][0]:block[i][0][2]]
        #BGR转HSV
        hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
        #根据阈值构建掩膜
        mask=cv2.inRange(hsv,lower,upper)
        #统计权值
        w1=0
        for m in mask:
            w1+=m/255

        w2=0
        for n in w1:
            w2+=n

        #选出最大权值的区域
        if w2>maxweight:
            maxindex=i
            maxweight=w2

def stretching(img):
    '''
    图像拉伸函数
    '''
    maxi=float(img.max())
    mini=float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]=(255/(maxi-mini)*img[i,j]-(255*mini)/(maxi-mini))

    return img

def allbinaryzation(img):
    '''
    二值化处理函数
    '''
    maxi=float(img.max())
    mini=float(img.min())

    x=maxi-((maxi-mini)/2)
    #二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret,thresh=cv2.threshold(img,x,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    #返回二值化后的黑白图像
    return thresh

def cut_license(afterimg,rect):
    '''
    图像分割函数
    '''
    #转换为宽度和高度
    rect[2]=rect[2]-rect[0]
    rect[3]=rect[3]-rect[1]
    rect_copy=tuple(rect.copy())#tuple是一个元组
    print("rect_copy",rect_copy)
    rect=[0,0,0,0]
    #创建掩膜
    mask=np.zeros(afterimg.shape[:2],np.uint8)
    #创建背景模型  大小只能为13*5，行数只能为1，单通道浮点型
    bgdModel=np.zeros((1,65),np.float64)
    #创建前景模型
    fgdModel=np.zeros((1,65),np.float64)
    #分割图像
    cv2.grabCut(afterimg,mask,rect_copy,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_show=afterimg*mask2[:,:,np.newaxis]
    return img_show

def lice_binarization(licenseimg):
    '''
    车牌图片二值化
    '''
    #车牌变为灰度图像
    gray_img=cv2.cvtColor(licenseimg,cv2.COLOR_BGR2GRAY)

    #均值滤波  去除噪声
    kernel=np.ones((3,3),np.float32)/9
    gray_img=cv2.filter2D(gray_img,-1,kernel)

    #二值化处理
    ret,thresh=cv2.threshold(gray_img,120,255,cv2.THRESH_BINARY)

    return thresh

def find_end(start,arg,black,white,width,black_max,white_max):
    end=start+1
    for m in range(start+1,width-1):
        if (black[m] if arg else white[m])>(0.95*black_max if arg else 0.95*white_max):
            end=m
            break
    return end

def find_rectangle(contour):
    '''
    寻找矩形轮廓
    '''
    y,x=[],[]

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y),min(x),max(y),max(x)]
