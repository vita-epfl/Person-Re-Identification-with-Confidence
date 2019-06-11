from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

import scipy.io as sio
import pdb
from collections import defaultdict
class CUB200_2011(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'CUB_200_2011'
    def __init__(self, root='data', verbose=True, **kwargs):
        super(CUB200_2011, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.imgs_dir = osp.join(self.dataset_dir, 'images')
        self.img_files = osp.join(self.dataset_dir, 'images.txt')
        self.bounding_file = osp.join(self.dataset_dir,'bounding_boxes.txt')
        self.train_test_file = osp.join(self.dataset_dir, 'train_test_split.txt')
        self.label_file = osp.join(self.dataset_dir, 'image_class_labels.txt')

        self._check_before_run()

        (train, num_train_pids, num_train_imgs), (test, num_test_pids, num_test_imgs) = self._process_dir()
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_test_imgs

        if verbose:
            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  test    | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.test = test

        self.num_train_pids = num_train_pids
        self.num_test_pids = num_test_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.imgs_dir):
            raise RuntimeError("'{}' is not available".format(self.imgs_dir))
        if not osp.exists(self.img_files):
            raise RuntimeError("'{}' is not available".format(self.img_files))
        if not osp.exists(self.train_test_file):
            raise RuntimeError("'{}' is not available".format(self.train_test_file))
        if not osp.exists(self.label_file):
            raise RuntimeError("'{}' is not available".format(self.label_file))

    def _process_dir(self):
        img_dict = defaultdict(list)
        with open(self.img_files, 'r') as file:
            for row in file:
                img_id, img_path = row.split()
                img_dict[img_id].append(osp.join(self.imgs_dir,img_path))

        train_pid_container = set()
        with open(self.label_file, 'r') as file:
            for row in file:
                img_id, img_label = row.split()
                img_dict[img_id].append(int(img_label))
                if int(img_label) <=100:
                    train_pid_container.add(int(img_label))
            pid2label = {class_id: label for label, class_id in enumerate(train_pid_container)}

        with open(self.bounding_file, 'r') as file:
            for row in file:
                img_id, x, y, width, height = row.split()
                img_dict[img_id].append(float(x))
                img_dict[img_id].append(float(y))
                img_dict[img_id].append(float(x)+float(width))
                img_dict[img_id].append(float(y)+float(height))
        train_dataset = []
        test_dataset = []
        test_pid_container = set()
        for data in img_dict.values():
            if int(data[1]) >100:
                test_dataset.append(tuple(data))
                test_pid_container.add(data[1])
                continue
            data[1]= pid2label[data[1]]
            train_dataset.append(tuple(data))

        num_train_pids = len(train_pid_container)
        num_train_imgs = len(train_dataset)

        num_test_pids = len(test_pid_container)
        num_test_imgs = len(test_dataset)
        return (train_dataset, num_train_pids, num_train_imgs), (test_dataset, num_test_pids, num_test_imgs)

    def get_testSet(self):
        return self.test, self.num_test_pids
