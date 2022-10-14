# Emojify表情识别



## 一、环境管理和package准备

### (一) 新建python3.6 报错

CondaError: Unable to create prefix directory 'D:\anaconda\envs\emojify'.
Check that you have sufficient permissions.

原因：`envs`文件夹需要管理员权限

找到envs文件夹，属性-安全-编辑-users-完全控制-确定

重新设置环境即可



### (二) package问题

#### 1.package下载

* conda install报错

```
Collecting package metadata (current_repodata.json): failed

CondaSSLError: OpenSSL appears to be unavailable on this machine. OpenSSL is required to
download and install packages.

Exception: HTTPSConnectionPool(host='conda.anaconda.org', port=443): Max retries exceeded with url: /conda-forge/win-64/current_repodata.json (Caused by SSLError("Can't connect to HTTPS URL because the SSL module is not available."))

```

缺少OpenSSL，去下一个，添加环境变量，还是不行

添加镜像源，还是不行

换成pip还是不行

关掉梯子pip就可以了



* 需要的package

​	pip install numpy

​	pip install opencv-python

​	pip install keras

​	pip install pillow

​	pip install SciPy，图片转换需要SciPy



#### 2.package修改

①

```python
from keras.emotion_models import Sequential
```

emotion_models和Sequential部分报错

新版本下应改为keras.models



②

```python
from keras.optimizers import Adam
```

报错，改为

```python
from keras.optimizer_v2.adam import Adam
```



③

cv2不报错，但是没有代码补全，cv2.xx报错

把opencv-python降级为4.5.4.60然后重启



#### 3.package调用补充

loss函数需要categorical_crossentropy

```python
from keras.losses import categorical_crossentropy
```





## 二、修改

### (一) 调用文件地址修改为本地的

* `train_dir `和 `val_dir` 地址修改成`'train'`和`'test'`
* `cv2.CascadeClassifier()`加载级联分类器，地址改为`'D:/anaconda/envs/emojify/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'`



### (二) 函数和参数名修改

* color_mode里gray的那个改成grayscale
* cv2.COLOR_BGR2gray_frame没了，改成cv2.cv2.COLOR_BGR2GRAY
* optimizer Adam的参数lr改成learning_rate
* emotion_model.fit_generator改成~.fit



### (三) 错误修改

* emotion_model.add后面多了一行load_weights,错误的，直接删除。load_weights应该是在模型训练好后经过save_weights后再来调用的。



### (四) GPU加速

用CPU跑一次要一个多小时 崩溃

安装TensorFlow，CUDA，cudNN 一一对应的的版本!(在TensorFlow官网可以查找对应关系)

用NVIDIA控制面板查看了我能安装的cuda到了v11.7，但TensorFlow最高对应11.2

最后选择安装TensorFlow2.6.0, CUDA11.2，cudNN8.1.0

pip install tensorflow==2.6.0

pip install tensorflow-gpu==2.6.0



安装CUDA选择自定义安装，默认C盘，取消勾选VS的选项



安装cudNN，是个压缩包，内容相当于TensorFlow中使用CUDA需要的补丁，解压出来之后要把对应目录的文件转移到CUDA对应目录下



运行后仍然报错缺失cudart等等的一堆文件，重启pycharm之后就不报错了



使用GPU加速，运行一次大概花二十几分钟，快了一些



### (五) 过拟合问题(未解决)

在dense层增加kernel_regularizer，使用l2正则化，结果验证集准确率最终还是没什么变化

减少了一两层conv2d的节点数为64，没用

减少dense节点为512，没啥用

我哭死，最后都改回去了

基本上epoch到第十轮之后，验证集val_acc稳定在61左右，而loss持续下降，train_acc保持小幅上升，acc差距逐渐增大



## 三、运行结果

### (一) 评价指标

![](https://s1.328888.xyz/2022/10/13/8Ubdg.png)

* loss: 0.3316
* train_accuracy: 0.8821
* val_accuracy:: 0.6251

### (二) 复现画面截图

<img src="https://s1.328888.xyz/2022/10/13/8IJVS.jpg" style="zoom: 25%;" /> <img src="https://s1.328888.xyz/2022/10/13/8Itp5.jpg" style="zoom:25%;" />

<img src="https://s1.328888.xyz/2022/10/13/8IzON.jpg" style="zoom:25%;" /> <img src="https://s1.328888.xyz/2022/10/13/8Phim.jpg" style="zoom:25%;" />

<img src="https://s1.328888.xyz/2022/10/13/8fSgs.jpg" style="zoom:25%;" /> <img src="https://s1.328888.xyz/2022/10/13/8fqHF.jpg" style="zoom:25%;" />

识别不出disgusted，QAQ



## 四、代码设计结构

### (一) 载入数据

训练集train和验证集test下载到本地目录

**数据增强**ImageDataGenerator()

1. 先用ImageDataGenerator(rescale=1./255)进行rescale将像素**重缩放**，使得像素值落在模型可以有效计算的范围内
2. 用其下的flow_from_directory()进一步**转化图像**
   * 尺寸统一为(48, 48)，target_size
   * 改成灰度图，有1个通道, color_mode="grayscale"
   * 分类算法返回label为独热码, class_mode="categorical"
   * batch_size=64

### (二) 设计trainer（训练器）

####  1.**反向传播**

model.compile() 

* loss损失：categorical_crossentropy多分类损失函数，搭配softmax食用哦
* optimizer优化：Adam优化器（设置lr和decay）
* metrics评价：acc准确率

#### 2.训练框架

model.fit()

* 放入训练集
* 设置轮数epochs
* 设置每轮训练的数据量steps_per_epoch，为训练集总量/batch_size
* 设置验证集validation_data
* 设置每轮验证的数据量validation_steps，为验证集总量/batch_size



### (三) 网络结构

* **模型**：顺序模型Sequential()

* **网络层**：

   卷积层-->卷积层-->池化层-->dropout层-->

   卷积层-->池化层-->卷积层-->池化层-->dropout层-->

   抹平层-->全连接层-->dropout层-->全连接层

  *  卷积层：用卷积核滑过图像处理，一个卷积核对应一种图像特征，处理后像素差值变大，特征显现，像素值越大特征越强
  * 池化层：使用最大池化，对每一小块区域取最大值，舍弃其他的，这样可以把维度中长宽缩小，而增加了高度，提取特征过滤信息
  * dropout层：使得这一层的节点中以输入的概率停止工作，这样可以一定程度上缓解过拟合（不让一些无关紧要的特征成为重要判断依据）
  * 抹平层：经过前几层之后，h变得比较大，flatten抹平为1个高度，以用于后面的全连接
  * 全连接层：把前一层所有节点连接起来，每个特征对应其权重，将特征汇总

  最后的全连接层节点为7，用softmax激活，得到概率分布。其他卷积层全连接层激活函数都用relu。





## 五、gui.py部分

### (一) 摄像头无法启用

cap1 = cv2.VideoCapture(0)从def里面改到最后主干中

有系统不兼容的warning，0后加上参数cv2.CAP_DSHOW就好了



### (二) 识别情况

<img src="https://s1.328888.xyz/2022/10/14/8zeod.png" style="zoom:22%;" /> <img src="https://s1.328888.xyz/2022/10/14/8zW0r.jpg" style="zoom:22%;" />

<img src="https://s1.328888.xyz/2022/10/14/8zvrm.jpg" style="zoom:22%;" /> <img src="https://s1.328888.xyz/2022/10/14/87Lm7.jpg" style="zoom:22%;" />

<img src="https://s1.328888.xyz/2022/10/14/87N1k.png" style="zoom:22%;" /> <img src="https://s1.328888.xyz/2022/10/14/87lyE.png" style="zoom:22%;" />

没有disgusted

QAQ
