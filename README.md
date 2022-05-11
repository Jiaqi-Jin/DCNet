## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __Detecting Sematic Lines Using Deep Chain Network__ 
## Network Architecture

![pipeline](https://github.com/pljq/DCNet/blob/main/pipeline.png)


## Require
Please `pip install` the following packages:
- Cython
- torch>=1.5
- torchvision>=0.6.1
- progress
- matplotlib
- scipy
- numpy
- opencv
- deep-hough

## Development Environment

Running on Ubuntu 18.04 system with pytorch 3.6, 8G VRAM.

## Inference
### step 1: Install python packages in [requirement.txt](https://github.com/pljq/DCNet/blob/main/requirements.txt) .

### step 2: Download the weight `model/Ours/paf_800X800_6000_80_14_8_SGD_0.1.pth` to the root directory.

- Model weights and test results download link：[af9p](https://pan.baidu.com/s/1coFL9CIx0wu7twu5fD9gog).

### step 3: Run the following code to test the image.
  `python test.py --image_path [your test image path] --model [pretrain model path] --save_path [save path]`

- Partial test results：

![part_res](https://github.com/pljq/DCNet/blob/main/part_res.png)

__Note: The pixels of the test image of this model are approximately `400 px X 400 px`__.

