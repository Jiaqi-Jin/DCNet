## Introduction

### PHNet——基于深度关联域和霍夫变换的语义直线检测算法

## Network Architecture

![pipeline](https://github.com/pljq/PHNet/blob/main/pipeline.png)

## Require
- 第三方Python包详见[requirement.txt](https://github.com/pljq/PHNet/blob/main/requirements.txt)

## Development Environment
- 使用pytorch 3.6 在Ubuntu 18.04系统上运行，显存8G

## Inference
### step 1: 安装 [requirement.txt](https://github.com/pljq/PHNet/blob/main/requirements.txt) 中的Python包

### step 2: 下载预训练模型

  - 预训练模型的下载链接：[dhls](https://pan.baidu.com/s/1JnNwUEcJK6opUg9yfqgiOg)

### step 3: 运行以下代码测试图片
  `python test.py --image_path [your test image path] --model [pretrain model path] --save_path [save path]`
- 部分测试结果：

![part_res](https://github.com/pljq/PHNet/blob/main/part_res.png)

__注意： 本模型测试图片像素约为`400px X 400px`__
