- 相关资源: [github](https://github.com/hexbo/baby-cv-learning)

# 第一课 计算机视觉与 OpenMMLab 开源算法体系 张子豪

## 计算机视觉基础

- 计算机视觉：让计算机理解图像、视频。
- 计算机视觉的三大基础任务：图像分类(图像识别)、目标检测、图像分割任务。
- 根据目标数量，计算机视觉任务也分为：单目标、多目标任务；
![](https://img-blog.csdnimg.cn/6a6b80a3239d48d8b307aaa94fd036c4.png)
![](https://img-blog.csdnimg.cn/23ca9307d4ce4d6c8726aa8a06c17eea.png)
![](https://img-blog.csdnimg.cn/d650a44103bf42c9949e31e75fd8cc58.png)
![](https://img-blog.csdnimg.cn/0b8535a4ea0645b9a6f2cb4520d9bd27.png)

- 图像分割又分为语义分割、实例分割: 语义分割不需要处理重合的情况，但实例分割需要。
- 大规模视觉识别挑战赛 ILSVRC SOTA 模型：AlexNet(2012) -> ZFNet(2013) -> GoogLeNet(2014) -> ResNet(2016) -> SENet(2017) -> ...
![](https://img-blog.csdnimg.cn/1082989904be43f7af2f51abc976a151.png)
- 计算机视觉具体应用场景举例：图像识别(识别照片中的物体是什么)、人脸检测与定位(特殊的识别和检测对象，用于支付、身份认证、换脸、虚拟主播)、姿态检测、自动驾驶、图像生成(GAN)与图像风格迁移、视频理解(自动剪辑、视频搜索)、文本生成图片、视觉大模型、神经渲染(NeRF 神经辐射场)..

## OpenMMLab 基础
- OpenMMLab 是基于 PyTorch 搭建的算法库，是深度学习用在计算机视觉方向的主流开源算法库（下图是1.0版本的架构图，目前正在迁移到2.0，2022年发布）。
![](https://img-blog.csdnimg.cn/8459214df107419e92d2242795b9aa58.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d589957a594f43ffa00cbf9533ff5676.png)



- MMDetection：目标检测、实例分割、全景分割(在实例分割基础上也对环境做感知)；
- MMDetection3D: MMDetection 用于 2D 数据，MMDetection3D 处理 3D 点云数据；
- MMClassification： 
![](https://img-blog.csdnimg.cn/c7fbea3646814249876777714b38d107.png)

- MMSegmentation：无人驾驶、遥感、医疗影像分析
![](https://img-blog.csdnimg.cn/8a5c18101df14083b01f569a2899e4a2.png)

- MMPose & MMHuman3D：人体姿态估计
- MMTracking：视频目标检测、单目标跟踪、多目标跟踪
- MMAction2：行为识别、时序动作检测、时空动作检测
![](https://img-blog.csdnimg.cn/78f755ada9854040b5691d8bbf8ad058.png)

- MMOCR：文本检测、文本识别、关键信息提取
- MMEditing：图像修复、抠图、超分辨率、图像生成

### OpenMMLab 2.0
更细节内容请看官方介绍或者repo中的pdf原文件。
![](https://img-blog.csdnimg.cn/f0e530ad90c042a0bd0765c35ab0a624.png)
![](https://img-blog.csdnimg.cn/481f36b4355743c4b9411118ea97be61.png)


# 机器学习和神经网络简介
这部分内容与计算机视觉无关，这里略掉，笔记默认读者有机器学习、深度学习基础概念，但没有系统学习计算机视觉。

