# The Forestry Pest Detection Algorithm in Complex Scenes Based on Adaptive Enhancement

## Overview
Abstract: Detection of forestry pests is crucial for protecting ecological balance and maintaining forest health. However, real-world pest monitoring scenarios in forests are often complex and diverse. Image quality degradation caused by rain, fog, occlusion, dust, low illumination, and camera defocus results in decreased detection network performance, posing challenges to traditional methods. Currently, no effective solution exists for pest detection in such complex scenarios. PCSNet (Forestry Pest Detection Network in Complex Scenarios), a YOLOv11-based network for forestry pest detection under challenging conditions, is proposed. A Chain-of-Thought Prompted Adaptive (CPA) enhancement module is introduced prior to the backbone to address unknown types of image quality degradation. Subsequently, Wavelet Transform Convolution (WTConv) is incorporated to mitigate over-parameterisation caused by increased receptive fields. Additionally, a lightweight downsampling module (ADown) is adopted to reduce model complexity. Based on these, PCSNet is designed for complex scene pest detection. Furthermore, combining pruning and distillation, a lightweight version PCSNet-Light is developed to meet embedded deployment needs. Experimental results demonstrate that PCSNet-Light achieves accuracy, recall, and mAP50 of 95.3%, 93.3%, and 96.6%, improving over the original model by 1.0%, 4.7%, and 1.5%, respectively, while reducing parameters by 0.5 million. Deployed on Raspberry Pi 5, it runs at 15.67 frames per second. With efficient performance on edge devices, this technology supports real-time forest pest monitoring, contributing to smarter and more sustainable forest management.

## Dataset
The dataset path and label settings are located in the 目标检测yolo/mydataset-for31.yaml.

## Train
The model training entry is located at 目标检测yolo/train.py.


update log:
- 低照度增强，若干网络实现
- yolo系列源码实现
- mmdetection实现
  