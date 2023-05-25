# Traffic-signal-recognition--
本项目使用YOLOv4模型，并在对数字信号灯进行数字识别时采用opencv算法。


## 环境安装

所需环境 python =3.7.11 torch==1.2.00

使用

```
pip install -r requirements.txt
```

安装所需的包。

## 文件下载

训练所需的预训练权重可在百度网盘中下载。 

链接：https://pan.baidu.com/s/1gKmRdwpQ05fMu1H-mi38zg 提取码：1234

作者训练结果可在下方链接中下载。

链接：https://pan.baidu.com/s/1cLSoWbra612Ezx1EsqOFGQ 提取码： 1234

## 训练过程

1.数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2.数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为： 

```python
左转红灯
左转绿灯
...
```

其中内容也可以换成自己需要的。

3. 开始网络训练  
   **训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
   **classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
   修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
   训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
   **model_path指向训练好的权值文件，在logs文件夹里。  
   classes_path指向检测类别所对应的txt。**  
   完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

5. 由于本项目不仅要对红绿灯进行识别，还要对倒计时识别，先采用CNN网络预先对数码管数据集进行训练。然后采用OpenCV对第一步预测出来的结果进行切割，然后把切割出来的图像进行二值化，再进行识别。

   

## 预测过程

在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。

```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolo_weights.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```

## 预测结果

![clip_image001](https://github.com/liwenchao956/Traffic-signal-recognition--/assets/86154097/292e6b32-53f4-4e24-8d30-01029dbb66e1)

![clip_image002](https://github.com/liwenchao956/Traffic-signal-recognition--/assets/86154097/b55a855c-727e-4d4b-9871-0654b4f4f814)

![clip_image002-16850168028411](https://github.com/liwenchao956/Traffic-signal-recognition--/assets/86154097/c2e961e1-e3ef-47f9-a07b-4f723ee53d95)

![clip_image002-16850168213472](https://github.com/liwenchao956/Traffic-signal-recognition--/assets/86154097/221497a3-b2ff-462c-a7e7-73488878d19c)

![clip_image002-16850168336033](https://github.com/liwenchao956/Traffic-signal-recognition--/assets/86154097/888baa73-99af-4d15-87df-015c7448fdd5)




## Reference

https://github.com/[bubbliiiing]
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  

