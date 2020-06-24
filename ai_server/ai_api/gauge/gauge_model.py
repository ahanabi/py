import tensorflow as tf
import os
import time
import numpy as np
import random
import cv2
import ai_api.utils.image_helpler as image_helpler


class GaugeModel():
    '''指针仪表识别模型'''
    # 静态对象
    StaticModel= None

    def GetStaticModel(model_path='./data/gauge_model'):
        if GaugeModel.StaticModel is None:
            GaugeModel.StaticModel = GaugeModel(model_path)
        return GaugeModel.StaticModel

    def __init__(self, model_path='./data/gauge_model'):
        # 设置GPU显存自适应
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        print(gpus, cpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) > 1:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        # 加载模型路径
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path
        # 建立模型
        self.build_model()
        # 加载模型
        self.load_model()

    def build_model(self):
        '''建立模型'''
        # 建立预测模型
        self.build_classes_model()
        # 优化器
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=3e-6)
        # self.optimizer = tf.keras.optimizers.SGD()
        # 损失函数
        # self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        # self.loss_object = tf.keras.losses.MeanAbsolutePercentageError()
        # 保存模型
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, 
            classes_model=self.classes_model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.model_path, max_to_keep=3)

    def conv_layer(self, input, filters, kernel_size, strides=(1, 1), padding='same'):
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides,
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def resnet_layer(self, input, filters, layer_sizes):
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input)
        x = self.conv_layer(x, filters, (3, 3),
                            strides=(2, 2), padding='valid')
        for _ in range(layer_sizes):
            x2 = x
            x = self.conv_layer(x, filters // 2, (1, 1))
            x = self.conv_layer(x, filters, (3, 3))
            x = tf.keras.layers.Add()([x, x2])
        return x

    def build_classes_model(self):
        '''建立预测模型'''
        # 所有参数
        input_classes = tf.keras.Input([400, 400, 3], dtype=tf.float32)
        x = tf.pad(input_classes, [[0, 0], [8, 8], [8, 8], [0, 0]], "CONSTANT")
        # (416 * 416)
        x = self.conv_layer(x, 32, (3, 3))
        # (208 * 208)
        x = self.resnet_layer(x, 64, 1)
        # (104 * 104)
        x = self.resnet_layer(x, 128, 2)
        # (52 * 52)
        x = self.resnet_layer(x, 256, 8)
        # (26 * 26)
        x = self.resnet_layer(x, 512, 8)
        y2 = x
        # (13 * 13)
        x = self.resnet_layer(x, 1024, 4)
        x = self.conv_layer(x, 512, (1, 1))
        x = self.conv_layer(x, 1024, (3, 3))
        x = self.conv_layer(x, 512, (1, 1))
        x = self.conv_layer(x, 1024, (3, 3))
        x = self.conv_layer(x, 512, (1, 1))
        x = self.conv_layer(x, 1024, (3, 3))
        x = tf.keras.layers.Conv2D(512, (1, 1), padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), use_bias=False)(x)
        x2 = tf.keras.layers.MaxPool2D((2, 2))(y2)
        x2 = tf.keras.layers.Conv2D(512, (1, 1), padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), use_bias=False)(x2)
        x = tf.keras.layers.Concatenate()([x, x2])
        x = tf.keras.layers.Conv2D(
            9, (13, 13), padding='valid', use_bias=False)(x)
        x = tf.keras.layers.Flatten()(x)
        self.classes_model = tf.keras.Model(inputs=input_classes, outputs=x)
    
    def get_random_data(self, image, value, target_points=None):
        '''生成随机图片与标签，用于训练'''
        # 画矩形
        # cv2.rectangle(image, (20, 20), (380, 380), tuple(np.random.randint(0, 30, (3), dtype=np.int32)), thickness=8)
        # 变换图像
        random_offset_x = random.random()*90-45
        random_offset_y = random.random()*90-45
        random_angle_x = random.random()*60-30
        random_angle_y = random.random()*60-30
        random_scale = random.random()*1.0+0.8
        # random_offset_x = 0
        # random_offset_y = 0
        # random_angle_x = 0
        # random_angle_y = 0
        # random_scale = 1
        # 点列表
        points = np.float32([[50, 50], # 左上
                            [50, 350], # 左下
                            [350, 50], # 右上
                            [350, 350]]) # 右下
        if target_points is None:
            target_points = points
        image, org, dst, perspective_points = image_helpler.opencvPerspective(image, offset=(random_offset_x, random_offset_y, 0),
                                                        angle=(random_angle_x, random_angle_y, 0), scale=(random_scale, random_scale, 1), points=target_points)
        # 计算四个角变换差值
        perspective_points = (perspective_points - points)/400
        # 增加噪声
        # image = image_helpler.opencvRandomLines(image, 8)
        image = image_helpler.opencvNoise(image)
        # 颜色抖动
        image = image_helpler.opencvRandomColor(image)

        # cv2.imwrite(path, image)

        # 最后输出图片
        random_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 调整参数范围
        random_img = random_img.astype(np.float32)
        random_img = random_img / 255
        value = value / 100
        # 标签
        # target_data = np.float32([[value]])
        # target_data = np.float32([[value, random_offset_x/400, random_offset_y/400,
        #                            random_angle_x/180, random_angle_y/180, random_scale]])
        target_data = np.float32([value, perspective_points[0,0], perspective_points[0,1],
                                    perspective_points[1,0], perspective_points[1,1],
                                    perspective_points[2,0], perspective_points[2,1],
                                    perspective_points[3,0], perspective_points[3,1]])
        # print('random_img:', random_img.shape)
        # print('target_data:', target_data.shape)
        return random_img, target_data
    
    def get_random_image(self, image):
        '''生成随机图片，用于测试'''
        # 画矩形
        # cv2.rectangle(image, (20, 20), (380, 380), tuple(np.random.randint(0, 30, (3), dtype=np.int32)), thickness=8)
        # 变换图像
        random_offset_x = random.random()*90-45
        random_offset_y = random.random()*90-45
        random_angle_x = random.random()*60-30
        random_angle_y = random.random()*60-30
        random_scale = random.random()*1.0+0.8
        # random_offset_x = 0
        # random_offset_y = 0
        # random_angle_x = 0
        # random_angle_y = 0
        # random_scale = 1
        random_img, org, dst, perspective_points = image_helpler.opencvPerspective(image, offset=(random_offset_x, random_offset_y, 0),
                                                        angle=(random_angle_x, random_angle_y, 0), scale=(random_scale, random_scale, 1))
        # 增加噪声
        # random_img = image_helpler.opencvRandomLines(random_img, 8)
        random_img = image_helpler.opencvNoise(random_img)
        # 颜色抖动
        random_img = image_helpler.opencvRandomColor(random_img)
        return random_img

    @tf.function
    def loss_fun(self, y_true, y_pred):
        value_loss = tf.math.reduce_sum(tf.math.abs(y_true[:,0]-y_pred[:,0]))
        value_p1 = tf.math.reduce_sum(tf.math.square(y_true[:,1:3]-y_pred[:,1:3]))
        value_p2 = tf.math.reduce_sum(tf.math.square(y_true[:,3:5]-y_pred[:,3:5]))
        value_p3 = tf.math.reduce_sum(tf.math.square(y_true[:,5:7]-y_pred[:,5:7]))
        value_p4 = tf.math.reduce_sum(tf.math.square(y_true[:,7:9]-y_pred[:,7:9]))
        loss = value_loss + (value_p1 + value_p2 + value_p3 + value_p4) / 8.0
        # tf.print('l1:', loss)
        # tf.print('l2:', tf.math.reduce_mean(tf.math.square(y_true-y_pred)))
        return loss

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, 400, 400, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 9), dtype=tf.float32),
    ))
    def train_step(self, input_image, target_data):
        '''
        单步训练
        input_image:图片(400,400,3)
        target_data:一个指针值(1)+透视变换2位移、2旋转、1缩放(5个值)
        '''
        print('Tracing with train_step', type(input_image), type(target_data))
        print('Tracing with train_step', input_image.shape, target_data.shape)
        loss = 0.0
        with tf.GradientTape() as tape:
            # 预测
            output_classes = self.classes_model(input_image)
            # 计算损失
            # loss = self.loss_object(y_true=target_data, y_pred=output_classes)
            loss = self.loss_fun(y_true=target_data, y_pred=output_classes)

        trainable_variables = self.classes_model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss

    def fit_generator(self, generator, steps_per_epoch, epochs, initial_epoch=1, auto_save=False):
        '''批量训练'''
        for epoch in range(initial_epoch, epochs+1):
            start = time.process_time()
            epoch_loss = 0
            for steps in range(1, steps_per_epoch+1):
                x, y = next(generator)
                # print('generator', x.shape, y.shape)
                loss = self.train_step(x, y)
                epoch_loss += loss
                print('\rsteps:%d/%d, epochs:%d/%d, loss:%0.4f'
                      % (steps, steps_per_epoch, epoch, epochs, loss), end='')
            end = time.process_time()
            print('\rsteps:%d/%d, epochs:%d/%d, %0.4f S, loss:%0.4f, epoch_loss:%0.4f'
                  % (steps, steps_per_epoch, epoch, epochs, (end - start), loss, epoch_loss))
            if auto_save:
                self.save_model()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=(None, 400, 400, 3), dtype=tf.float32),
    ))
    def predict(self, input_image):
        '''
        预测(编译模式)
        input_image:图片(400,400,3)
        return:两个指针值(2)
        '''
        # 预测
        start = time.process_time()
        output_classes = self.classes_model(input_image)
        end = time.process_time()
        tf.print('%s predict time: %f' % (self.__class__, (end - start)))
        return output_classes

    def save_model(self):
        '''保存模型'''
        save_path = self.checkpoint_manager.save()
        print('保存模型 {}'.format(save_path))

    def load_model(self):
        '''加载模型'''
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('加载模型 {}'.format(self.checkpoint_manager.latest_checkpoint))


def main():
    model = GaugeModel()
    input_image = tf.random.uniform(
        [1, 400, 400, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)
    target_data = tf.random.uniform(
        [1, 2], minval=0, maxval=1, dtype=tf.dtypes.float32)
    model.train_step(input_image, target_data)
    pass


if __name__ == '__main__':
    main()
