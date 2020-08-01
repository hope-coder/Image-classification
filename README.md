# 使用Pytorch框架通过神经网络实现的图像二分类问题
  ## 问题描述
    本次实验使用多种算法实现了二分类问题，使用两个数据集对网络进行训练与学习。
  ## 数据集说明
    数据集一：
      纹理图片数据集，数据集来源：公开的纹理数据集。 百度云下载地址：https://pan.baidu.com/s/1NiFmTGCkTSCyNMLhsUln3w  提取码：z6hy 图片下载后解压在程序当前目录下即可
    数据集二：
      猫狗图片数据集，数据集来源：百度数据集。 下载地址：https://pan.baidu.com/s/1B01mUaodlPwdxBU-dLjkEg  提取码：dp74 图片下载后解压在程序当前目录下即可
  ## 代码说明
    此代码使用Pytorch框架，建立CNN卷积全连接神经网络以及ResNet网络。使用两种网络对数据进行训练以及测试，比较性能
    CNN.py:定义了CNN模型，定义了训练函数以及测试函数
    ResNet.py:定义了ResNet模型，定义了训练函数以及测试函数
    work.py：主程序，调用函数对数据进行训练以及测试
  ## 运行说明
    下载数据集到程序运行目录，解压。
    安装pytorch框架，该程序使用的是cpu版本，安装命令请访问https://pytorch.org/ ，若使用GPU版本请修改代码
    运行work.py
  
  
