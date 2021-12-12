import argparse
import os
import time
import sys
sys.path.append('..')
import torch
from lib.network.rtpose_vgg import get_model
import torch
import torch.optim
import numpy as np
import cv2
from skimage.measure import label, regionprops
from utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point

def construct_model(args):
    # model = pose_estimation.PoseModel(args)
    model = get_model(trunk='vgg19', args=args)# resnet50
    print(model)
    state_dict = torch.load(args.model)['model']
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda().float()
    model.eval()

    return model

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def normalize(origin_img):
    image = origin_img.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    image = preprocessed_img.astype(np.float32)
    return image

def process(model,  args):
    stride = args.stride
    padValue = args.padValue
    scale_search = args.scale_search
    boxsize = args.boxsize
    save_path = args.save_path
    input_image_path = args.image_path

    for file in os.listdir(input_image_path):
        input_path = input_image_path+"/"+file
        origin_img = cv2.imread(input_path)
        normed_img = normalize(origin_img)
        height, width, _ = normed_img.shape
        multiplier = [x * boxsize / height for x in scale_search]#diffenrent scale
        for scale_id in range(args.num_fpn_scale):
            for m in range(len(multiplier)):
                scale = multiplier[m]
                imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                imgToTest_padded, pad = padRightDownCorner(imgToTest, stride, padValue)
                input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
                input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda().float())
                tic = time.time()
                # get the features
                out_scales = model(input_var)
                toc = time.time()
                key_points = torch.sigmoid(out_scales[scale_id][-1])
                binary_kmap = key_points.squeeze().cpu().detach().numpy()>0.1

                kmap_label = label(binary_kmap, connectivity=1)
                props = regionprops(kmap_label)

                plist = []
                for prop in props:
                    plist.append(prop.centroid)

                # size = (size[0][0], size[0][1])
                b_points = reverse_mapping(plist, numAngle=100, numRho=50,
                                           size=(400, 400))
                scale_w = 1
                scale_h = 1
                for i in range(len(b_points)):
                    y1 = int(np.round(b_points[i][0] * scale_h))
                    x1 = int(np.round(b_points[i][1] * scale_w))
                    y2 = int(np.round(b_points[i][2] * scale_h))
                    x2 = int(np.round(b_points[i][3] * scale_w))
                    if x1 == x2:
                        angle = -np.pi / 2
                    else:
                        angle = np.arctan((y1 - y2) / (x1 - x2))
                    (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, 400, 400)
                    b_points[i] = (y1, x1, y2, x2)

                # vis = visulize_mapping(b_points, size[::-1], names[0])
                for (y1, x1, y2, x2) in b_points:
                    img = cv2.line(origin_img, (x1, y1), (x2, y2), (255, 255, 0), thickness=int(0.01 * max(400, 400)))

                cv2.imwrite(save_path + file, img)
        print(file[:4] + ' ' + 'processing time is %.5f' % (toc - tic))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='input image')
    parser.add_argument('--model', type=str, required=True, help='path to the weights file')
    parser.add_argument('--save_path', type=str, required=True, help='path to save results')
    parser.add_argument('--boxsize', type=int, default=400)
    parser.add_argument('--scale-search', type=int, default=[1])
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padValue', type=int, default=0)
    parser.add_argument('--num_fpn_scale', type=int, default=1)

    args = parser.parse_args()

    # load model
    model = construct_model(args)

    print('start processing...')

    process(model, args)
