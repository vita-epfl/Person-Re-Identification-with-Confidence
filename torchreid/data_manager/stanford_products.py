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
class StanforOnlineProducts(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Stanford_Online_Products'
    def __init__(self, root='data', verbose=True, **kwargs):
        super(StanforOnlineProducts, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_file = osp.join(self.dataset_dir, 'Ebay_train.txt')
        self.test_file = osp.join(self.dataset_dir, 'Ebay_test.txt')

        self._check_before_run()

        (train, num_train_pids, num_train_imgs)= self._process_dir(self.dataset_dir, self.train_file, relabel =True)
        (test, num_test_pids, num_test_imgs)  = self._process_dir(self.dataset_dir, self.test_file, relabel = False)
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
        if not osp.exists(self.train_file):
            raise RuntimeError("'{}' is not available".format(self.train_file))
        if not osp.exists(self.test_file):
            raise RuntimeError("'{}' is not available".format(self.test_file))

    def _process_dir(self, dir_path, file_path, relabel=False):
        dataset = []
        pid_container = set()
        with open(file_path, 'r') as file:
            for i, row in enumerate(file):
                if i == 0:
                    continue
                _, class_id, _, _ = row.split()
                pid_container.add(int(class_id))
            pid2label = {class_id: label for label, class_id in enumerate(pid_container)}
            file.seek(0)
            for i, row in enumerate(file):
                if i == 0:
                    continue
                image_id, class_id, _, img_path = row.split()
                img_path = osp.join(self.dataset_dir, img_path)
                if relabel: class_id = pid2label[int(class_id)]
                dataset.append((img_path, int(class_id)))
        num_pids = len(pid_container)
        num_imgs = len(dataset)

        return (dataset, num_pids, num_imgs)

    def get_testSet(self):
        return self.test, self.num_test_pids
