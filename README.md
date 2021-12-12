## Introduction

### PHNet——基于深度关联域和霍夫变换的语义直线检测算法

## Network Architecture

![pipeline](https://github.com/pljq/PHNet/pipeline.png)

## Require
- 第三方Python包详见[requirement.txt](https://github.com/pljq/PHNet/requirement.txt)

## Development Environment
- 使用pytorch 3.6 在Ubuntu 18.04系统上运行，显存8G

## Inference
### step 1: 安装 [requirement.txt](https://github.com/pljq/PHNet/requirement.txt) 中的Python包

### step 2: 下载预训练模型

  - 预训练模型下载链接：[8s1d](https://pan.baidu.com/s/1iRkM4wJckckfvb4vC6w8tQ)

### step 3: 运行以下代码测试图片
  `python test.py --image_path [your test image path] --model [pretrain model path] --save_path [save path]`
- 测试结果：

![0498](https://github.com/pljq/PHNet/img/results/0498.jpg)

![1173](https://github.com/pljq/PHNet/img/results/1173.jpg)

![1648](https://github.com/pljq/PHNet/img/results/1638.jpg)

__注意： 本模型测试图片像素约为`400px X 400px`__
