import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizer_v2.adam import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy

train_dir = 'train' # 更改了目录地址
val_dir = 'test' # 更改了目录地址
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
# ImageDataGenerator()是图像生成器，对批次中的图像进行数据增强处理
# rescale是重缩放，把每个像素的值乘上rescale，在变换操作的最前面执行，作用是收敛模型以确保落在激活函数的有效范围内，一般设置缩放因子为1/255

train_generator = train_datagen.flow_from_directory(
        train_dir, #训练集目录地址
        target_size=(48, 48), # 图像尺寸转换目标
        batch_size=64, # 每个批次生成图像数量为64，默认值是32
        color_mode="grayscale", # 新版名字已经变了，修改成grayscale，1颜色通道，默认值是rgb，3颜色通道
        class_mode='categorical') # 返回label标签数组类型，默认categorical是2D one-hot

validation_generator = val_datagen.flow_from_directory(
        val_dir, # 验证集
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = Sequential() # 顺序模型

# filters卷积核（滤波器），每一个扫过图像之后生成对应的一张图，对应一个特征，最后输出也就对应多少层，一般数量按2的幂递增到1024
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1))) # kernel_size卷积核大小,filters=32卷积核数量，input_shape(长宽频道)
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # 后面层不需要用到input_shape
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))   # 最大池化，提取特征，过滤信息
emotion_model.add(Dropout(0.25)) # 随机临时删除部分神经元，缓解过拟合

# 逐渐增多
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten()) # 抹平层，把高度压缩为1，以用于全连接
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax')) # 全连接层，分类七种，输出节点7，softmax激活得到概率

cv2.ocl.setUseOpenCL(False) #禁用，避免和CUDA冲突

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"} # 表情字典

emotion_model.compile(
    loss=categorical_crossentropy, # 修改了loss，去掉引号。多分类损失函数独热码
    optimizer=Adam( # 优化器Adam
        learning_rate=0.001, # lr改成learning_rate
        decay=1e-5), # 每次学习率下降多少
    metrics=['accuracy']) # 评价函数accuracy)

emotion_model_info = emotion_model.fit( #fit_generator改成fit
        train_generator, # 对象为训练集
        steps_per_epoch=28709 // 64, # 取整除,每个epoch训练的数量
        epochs=50, # 训练50轮
        validation_data=validation_generator, # 验证集
        validation_steps=7178 // 64) #每轮验证数量
# batch_size未指定则默认为32
emotion_model.save_weights('emotion_model.h5') # 把权重保存在emotion_model.h5文件中

# start the webcam feed
cap = cv2.VideoCapture(0) #0为打开摄像头 (不是显示窗口)，参数为路径则打开对应视频
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read() # 按帧读取视频，ret布尔变量，读到帧就是true，结束是false，frame是读取到的帧的三维矩阵
    if not ret:
        break
        # 读取失败就退出
    bounding_box = cv2.CascadeClassifier('D:/anaconda/envs/emojify/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') # 加载级联分类器haarcascade
    gray_frame = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2GRAY) # 修改为cv2.cv2.COLOR_BGR2GRAY。颜色转换cvtColor(图像，转换格式)默认格式是BGR，转成GRAY
    num_faces = bounding_box.detectMultiScale( #检测人脸函数，用到haarcascade，工作时滑动窗口来检测
        gray_frame, #摄像头捕捉转换后的灰度图
        scaleFactor=1.3, #像素缩小了多少，越大缩越快也越容易忽略某种大小
        minNeighbors=5) #相邻邻居也就是重叠的矩阵框，最小要检测到这个数量才算识别为人脸
    #返回 x,y坐上坐标, w,h矩阵宽高

    # 接下来要在脸上画框
    for (x, y, w, h) in num_faces:
        cv2.rectangle( #画矩形函数，注意参数顺序
            frame, # 图片
            (x, y-50), #左上角坐标
            (x+w, y+h+10), #右下角坐标
            (255, 0, 0), #画框颜色(B, G, R)
            2) #画框粗细
        roi_gray_frame = gray_frame[y:y + h, x:x + w] #利用num_faces返回值xywh对灰度图切片，roi表示RegionOfInterest想要的
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0) #expand_dims升维，axis即新增一维的位置
        emotion_prediction = emotion_model.predict(cropped_img) #放进模型预测
        maxindex = int(np.argmax(emotion_prediction)) #最大概率对应的表情号
        cv2.putText( #识别结果
            frame, #图片
            emotion_dict[maxindex], #写的内容，最大概率表情
            (x+20, y-60), #位置
            cv2.FONT_HERSHEY_SIMPLEX, #字体类型
            1, #字体大小
            (255, 255, 255), #字体颜色
            2, #字体粗细
            cv2.LINE_AA) # linetype抗锯齿线

    cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC)) #显示Video窗口，再把原图像改尺寸
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        # 按q退出

cap.release() # 关摄像头
cv2.destroyAllWindows() #释放相关的内存
