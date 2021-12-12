import copy
import logging
import os
import numpy as np
from numpy import ma
import torch.utils.data
import torchvision
from PIL import Image
import pylab
from os.path import join, split, isdir, isfile, abspath
import math
import cv2
import matplotlib.pyplot as plt
from .heatmap import putGaussianMaps
from .paf import putVecMaps
from . import transforms, utils
CocoPairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13) , (14, 15), (16, 17)]



def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('First_point'), keypoints.index('Second_point')],
        [keypoints.index('Second_point'), keypoints.index('Third_point')],
        [keypoints.index('Third_point'), keypoints.index('Fourth_point')],
        [keypoints.index('Fourth_point'), keypoints.index('Fifth_point')],
        [keypoints.index('Fifth_point'), keypoints.index('Sixth_point')],
        [keypoints.index('Sixth_point'), keypoints.index('Seventh_point')],
        [keypoints.index('Seventh_point'), keypoints.index('Eighth_point')],
        [keypoints.index('Eighth_point'), keypoints.index('Ninth_point')],
    ]
    return kp_lines
    
def get_keypoints():
    keypoints = [
        'First_point',
        'Second_point',
        'Third_point',
        'Fourth_point',
        'Fifth_point',
        'Sixth_point',
        'Seventh_point',
        'Eighth_point',
        'Ninth_point',
    ]

    return keypoints
    


class CocoKeypoints(torch.utils.data.Dataset):


    def __init__(self, root, annFile, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False,
                 input_y=400, input_x=400, stride=[8]):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=['Line'])
        if all_images:
            self.ids = self.coco.getImgIds()
        elif all_persons:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms

        # self.data_path = [join(root_dir, i + ".npy") for i in lines]
        self.HEATMAP_COUNT = len(get_keypoints())
        self.LIMB_IDS = kp_connections(get_keypoints())
        self.input_y = input_y
        self.input_x = input_x        
        self.stride = stride
        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... done.')

    def __getitem__(self, index):
        image_id = self.ids[index]
        # cv2.setNumThreads(0)
        # plt.figure(figsize(image_id)
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')
            f.close()

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        image, anns, meta = self.preprocess(image, anns, None)

        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta, meta_init)

        return self.single_image_processing(image, anns, meta, meta_init)

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size  #800,800


        image = self.image_transform(image)  # 3,400,400
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        heatmaps_scale, pafs_scale = self.get_ground_truth(anns)  # 多尺度特征图


        data_path = self.root + "/" + meta_init['file_name'][:-4] + ".npy"
        data = np.load(data_path, allow_pickle=True).item()
        hough_space_label8 = data["hough_space_label8"].astype(np.float32)
        gt_coords = data["coords"]
        hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0).cuda()

        heatmaps_dir = {}
        pafs_dir = {}
        for i in range(len(heatmaps_scale)):
            heatmaps_dir[i] = torch.from_numpy(heatmaps_scale[i].transpose((2, 0, 1)).astype(np.float32))
            pafs_dir[i] = torch.from_numpy(pafs_scale[i].transpose((2, 0, 1)).astype(np.float32))

        return anns[0]["image_id"], image, heatmaps_dir, pafs_dir, hough_space_label8, gt_coords

    def remove_illegal_joint(self, keypoints):#reduce

        MAGIC_CONSTANT = (-1, -1, 0)
        mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                     keypoints[:, :, 0] < 0,
                                     keypoints[:, :, 1] >= self.input_y,
                                     keypoints[:, :, 1] < 0))
        keypoints[mask] = MAGIC_CONSTANT

        return keypoints#qudia
        


    def get_ground_truth(self, anns):
        heatmaps_scale = []
        pafs_scale = []
        hough_space_label8_scale = []
        gt_coords_scale = []
        for idx, std in enumerate(self.stride):
            grid_y = math.ceil(self.input_y / std)  # 50
            grid_x = math.ceil(self.input_x / std)  # 50
            channels_heat = (self.HEATMAP_COUNT + 1)  # 6
            channels_paf = 2 * len(self.LIMB_IDS)  # 10
            heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
            pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))
            keypoints = []
            for ann in anns:
                single_keypoints = np.array(ann['keypoints']).reshape(-1,3)
                # single_keypoints = self.add_neck(single_keypoints)
                keypoints.append(single_keypoints)
            keypoints = np.array(keypoints)#(n,5,3)
            keypoints = self.remove_illegal_joint(keypoints)


            for i in range(self.HEATMAP_COUNT):
                joints = [jo[i] for jo in keypoints]
                for joint in joints:
                    if joint[2] > 0.5:
                        center = joint[:2]
                        gaussian_map = heatmaps[:, :, i]
                        heatmaps[:, :, i] = putGaussianMaps(
                            center, gaussian_map,
                            14.0, grid_y, grid_x, std)

            # pafs
            for i, (k1, k2) in enumerate(self.LIMB_IDS):
                # limb
                count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32) # 50, 50
                for joint in keypoints:#5,3
                    if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                        centerA = joint[k1, :2]
                        centerB = joint[k2, :2]
                        vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                        pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                            centerA=centerA,
                            centerB=centerB,
                            accumulate_vec_map=vec_map,
                            count=count, grid_y=grid_y, grid_x=grid_x, stride=std)

            # background
            heatmaps[:, :, -1] = np.maximum(
                1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
                0.)
            # data_path = self.root + "/" + meta_init['file_name'][:-4] + ".npy"
            # data = np.load(data_path, allow_pickle=True).item()
            # hough_space_label8 = data["hough_space_label8"].astype(np.float32)
            # gt_coords = data["coords"]
            # hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0).cuda()

            heatmaps_scale.append(heatmaps)
            pafs_scale.append(pafs)
            # hough_space_label8_scale.append(hough_space_label8)
            # gt_coords_scale.append(gt_coords)


        return heatmaps_scale, pafs_scale
        
    def __len__(self):
        return len(self.ids)


# class ImageList(torch.utils.data.Dataset):
#     def __init__(self, image_paths, preprocess=None, image_transform=None):
#         self.image_paths = image_paths
#         self.image_transform = image_transform or transforms.image_transform
#         self.preprocess = preprocess
#
#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         with open(image_path, 'rb') as f:
#             image = Image.open(f).convert('RGB')
#
#         if self.preprocess is not None:
#             image = self.preprocess(image, [], None)[0]
#
#         original_image = torchvision.transforms.functional.to_tensor(image)
#         image = self.image_transform(image)
#
#         return image_path, original_image, image
#
#     def __len__(self):
#         return len(self.image_paths)
#
#
# class PilImageList(torch.utils.data.Dataset):
#     def __init__(self, images, image_transform=None):
#         self.images = images
#         self.image_transform = image_transform or transforms.image_transform
#
#     def __getitem__(self, index):
#         pil_image = self.images[index].copy().convert('RGB')
#         original_image = torchvision.transforms.functional.to_tensor(pil_image)
#         image = self.image_transform(pil_image)
#
#         return index, original_image, image
#
#     def __len__(self):
#         return len(self.images)

