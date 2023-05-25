# 训练+测试


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage import morphology

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# # 超参数
# EPOCH = 100  # 训练整批数据的次数
# BATCH_SIZE = 50
# LR = 0.001  # 学习率
# DOWNLOAD_MNIST = False  # 表示还没有下载数据集，如果数据集下载好了就写False
#
# # 下载mnist手写数据集
# train_data = torchvision.datasets.MNIST(
#     root='./data/',  # 保存或提取的位置  会放在当前文件夹中
#     train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
#     transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
#
#     download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
# )
#
# test_data = torchvision.datasets.MNIST(
#     root='./data/',
#     train=False  # 表明是测试集
# )
#
# # 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
# # Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
# train_loader = Data.DataLoader(
#     dataset=train_data,
#     batch_size=BATCH_SIZE,
#     shuffle=True  # 是否打乱数据，一般都打乱
# )
#
# # 进行测试
# # 为节约时间，测试时只测试前2000个
# #
# test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
# test_y = test_data.test_labels[:2000]


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出

class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类
        # self.out = nn.Linear(32 * 14 * 14, 10)  # 输出是10个类  输入56

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output


cnn = CNN()
# print(cnn)

# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

# # 优化器选择Adam
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# # 损失函数
# loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

def train():
    # 超参数
    EPOCH = 100  # 训练整批数据的次数
    BATCH_SIZE = 50
    LR = 0.001  # 学习率
    DOWNLOAD_MNIST = False  # 表示还没有下载数据集，如果数据集下载好了就写False

    # 下载mnist手写数据集
    train_data = torchvision.datasets.MNIST(
        root='./data/',  # 保存或提取的位置  会放在当前文件夹中
        train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
        transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

        download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
    )

    test_data = torchvision.datasets.MNIST(
        root='./data/',
        train=False  # 表明是测试集
    )

    # 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
    # Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 是否打乱数据，一般都打乱
    )

    # 进行测试
    # 为节约时间，测试时只测试前2000个
    #
    test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
    # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
    # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
    test_y = test_data.test_labels[:2000]
    # 训练
    # 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

    # 优化器选择Adam
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted


    #开始训练
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
            output = cnn(b_x)  # 先将数据放到cnn中计算output
            loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        if epoch % 10 == 0:
            torch.save(cnn.state_dict(), 'cnn2.pkl')#保存模型

    # 加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
    cnn.load_state_dict(torch.load('cnn2.pkl'))
    cnn.eval()
    # print 10 predictions from test data
    inputs = test_x[:32]  # 测试32个数据
    test_output = cnn(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')  # 打印识别后的数字
    # print(test_y[:10].numpy(), 'real number')

    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)

    # 下面三行为改变图片的亮度
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean
    cv2.imshow('win', img)  # opencv显示需要识别的数据图片
    key_pressed = cv2.waitKey(0)

def test(gray_img, index=None, imgname=None):
    # 加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
    input = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('./analysis/%s_%d_resize.jpg' %(imgname, index), input)

    # cv2.imwrite('resize.jpg', input)
    inputs = torch.unsqueeze(torch.FloatTensor(input), dim=0).type(torch.FloatTensor) / 255
    inputs = torch.unsqueeze(inputs, dim=0)

    # cnn.load_state_dict(torch.load('./number_recognition/cnn2.pkl'))
    cnn.load_state_dict(torch.load('./number_recognition/cnn2_number.pth'))
    cnn.eval()
    # print 10 predictions from test data
    # inputs = test_x[:32]  # 测试32个数据
    test_output = cnn(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    # print(pred_y, 'prediction number')  # 打印识别后的数字
    # print(test_y[:10].numpy(), 'real number')

    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)

    # 下面三行为改变图片的亮度
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean
    # cv2.imshow('win', img)  # opencv显示需要识别的数据图片
    # key_pressed = cv2.waitKey(0)
    return pred_y

def main_test(gray_img, imgname):
    # gray_img_path = './inference/test3.png_gray.png'
    # gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

    meanvalue = gray_img.mean()
    if meanvalue >= 200:
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        # plt.hist(hist.ravel(), 256, [0,256])
        # plt.savefig(filename + "_hist.png")
        # plt.show()
        min_val, max_val, min_index, max_index = cv2.minMaxLoc(hist)
        ret, image_bin = cv2.threshold(gray_img, int(max_index[1]) - 7, 255,
                                       cv2.THRESH_BINARY)
    else:
        mean, stddev = cv2.meanStdDev(gray_img)
        ret, image_bin = cv2.threshold(gray_img, meanvalue + 65, 255,
                                       cv2.THRESH_BINARY)

        # image_bin = cv2.adaptiveThreshold(image_gray, 255,
        #                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                  cv2.THRESH_BINARY, 11,
        #                                  0)

    cv2.imwrite('./analysis/%s_image_bin.jpg'%imgname, image_bin)
    kernel = np.ones((3, 3), np.int8)
    image_dil = cv2.dilate(image_bin, kernel, iterations=1)
    cv2.imwrite('./analysis/%s_image_dil.jpg'%imgname, image_dil)

    contours, hierarchy = cv2.findContours(image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRect = []
    predict_text = ''
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if h / w > 1:
            # boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_dil, (x, y), (x + w, y + h), 255, 2)
            area = cv2.contourArea(c)
            print("area: {}  imagename: {}".format(area, imgname))
            print("w*h: {}  imagename: {}".format(w*h, imgname))
            # if area > 10:  # 选择大于10000的面积
            if w*h >= 80:  # 选择大于10000的面积
                boundRect.append([x, y, w, h])

                height = image_dil.shape[0]
                width = image_dil.shape[1]
                ymin, ymax = y - 2, y + h + 2
                xmin, xmax = x - 2, x + w + 2
                if xmin < 0:
                    xmin = 0
                if xmax > width:
                    xmax = width
                if ymin < 0:
                    ymin = 0
                if ymax > height:
                    ymax = height

                roi = image_dil[ymin: ymax, xmin:xmax]
                cv2.imwrite('./analysis/%s_%d_roi.jpg' %(imgname, i), roi)

                # roi[roi == 255] = 1
                # skeleton0 = morphology.skeletonize(roi)
                # skeleton = skeleton0.astype(np.uint8) * 255
                # cv2.imwrite('%d_thin.jpg' % i, skeleton)

                predict = test(roi, index=i, imgname=imgname)
                if len(boundRect) != 0:
                    if x > boundRect[0][0]:
                        predict_text += str(predict[0])
                    else:
                        predict_text = str(predict[0]) + predict_text
                else:
                    predict_text += str(predict[0])
    if len(predict_text) > 2:
        predict_text = predict_text[0:2]
    print('prediction number: ', predict_text)
    return predict_text

def tomygray(image, labelflag):
    height = image.shape[0]
    width = image.shape[1]
    gray = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            # pixel = max(image[i,j][0], image[i,j][1], image[i,j][2])
            if labelflag == '红色倒计时':
                pixel = 0.0 * image[i, j][0] + 0.0 * image[i, j][1] + 1 * image[i, j][2]
            elif labelflag == '绿色倒计时':
                pixel = 0.0 * image[i, j][0] + 1.0 * image[i, j][1] + 0.0 * image[i, j][2]
            # pixel = 0.0 * image[i, j][0] + 0.3 * image[i, j][1] + 0.7 * image[i, j][2]
            gray[i, j] = pixel
    return gray

if __name__ == "__main__":
    # train()




    gray_img_path = './inference/test3.png_gray.png'
    gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

    meanvalue = gray_img.mean()
    if meanvalue >= 200:
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        # plt.hist(hist.ravel(), 256, [0,256])
        # plt.savefig(filename + "_hist.png")
        # plt.show()
        min_val, max_val, min_index, max_index = cv2.minMaxLoc(hist)
        ret, image_bin = cv2.threshold(gray_img, int(max_index[1]) - 7, 255,
                                       cv2.THRESH_BINARY)
    else:
        mean, stddev = cv2.meanStdDev(gray_img)
        ret, image_bin = cv2.threshold(gray_img, meanvalue + 65, 255,
                                       cv2.THRESH_BINARY)

        # image_bin = cv2.adaptiveThreshold(image_gray, 255,
        #                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                  cv2.THRESH_BINARY, 11,
        #                                  0)

    kernel = np.ones((3, 3), np.int8)
    image_dil = cv2.dilate(image_bin, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRect = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if h / w > 1:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_dil, (x, y), (x + w, y + h), 255, 2)

            height = image_dil.shape[0]
            width = image_dil.shape[1]
            ymin, ymax = y-5, y+h+5
            xmin, xmax = x-5, x+w+5
            if xmin < 0:
                xmin = 0
            if xmax > width:
                xmax = width
            if ymin < 0:
                ymin = 0
            if ymax > height:
                ymax = height

            roi = image_dil[ymin: ymax, xmin:xmax]
            cv2.imwrite('%d_roi.jpg' % i, roi)

            roi[roi == 255] = 1
            skeleton0 = morphology.skeletonize(roi)
            skeleton = skeleton0.astype(np.uint8) * 255
            cv2.imwrite('%d_thin.jpg'%i, skeleton)

            predict = test(skeleton, index=i)
            print('prediction number: ', predict)