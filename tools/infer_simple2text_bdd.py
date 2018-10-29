from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import json

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.vis import convert_from_cls_format
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
bdd_category = ['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider']

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)
    parser.add_argument(
        '--name', type=str, required=True, help='The name of the output file')
    parser.add_argument(
        '--width', type=int, default=1280, help='The name of the output file')
    parser.add_argument(
        '--height', type=int, default=720, help='The name of the output file')

    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    prefix_path = args.output_dir

    os.makedirs(prefix_path, exist_ok=True)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()
    if args.image_dir:
        imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images
    num_images = len(imglist)

    writen_results = []

    # validate
    demo_im = cv2.imread(imglist[0])
    print(np.shape(demo_im))
    h, w, _ = np.shape(demo_im)
    #print(h)
    #print(args.height)
    assert h == args.height
    assert w == args.width
    h_scale = 1280 / args.height
    w_scale = 720 / args.width

    for i in tqdm(range(num_images)):
        im = cv2.imread(imglist[i])
        assert im is not None

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)

        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))

        # boxs = [[x1, y1, x2, y2, cls], ...]
        boxes, _, _, classes = convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

        if boxes is None:
            continue
        # scale
        boxes[:, 0] = boxes[:, 0] * w_scale
        boxes[:, 2] = boxes[:, 2] * w_scale
        boxes[:, 1] = boxes[:, 1] * h_scale
        boxes[:, 3] = boxes[:, 3] * h_scale

        if classes == []:
            continue

        for instance_idx, cls_idx in enumerate(classes):
            cls_name = dataset.classes[cls_idx]
            if cls_name == 'motorcycle':
                cls_name = 'motor'
            elif cls_name == 'stop sign':
                cls_name = 'traffic sign'
            elif cls_name == 'bicycle':
                cls_name = 'bike'
            if cls_name not in bdd_category:
                continue

            writen_results.append({"name": imglist[i].split('/')[-1],
                                   "timestamp": 1000,
                                   "category": cls_name,
                                   "bbox": boxes[instance_idx, :4],
                                   "score": boxes[instance_idx, -1]})

    with open(os.path.join(prefix_path, args.name + '.json'), 'w') as outputfile:
        json.dump(writen_results, outputfile, cls=MyEncoder)


if __name__ == '__main__':
    main()
