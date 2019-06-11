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
class Cars196(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'cars196'
    def __init__(self, root='data', verbose=True, **kwargs):
        super(Cars196, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.im_dir = osp.join(self.dataset_dir, 'car_ims')
        self.mat_file = osp.join(self.dataset_dir, 'cars_annos.mat')

        self._check_before_run()

        (train, num_train_pids, num_train_imgs), (test, num_test_pids, num_test_imgs) = self._process_dir(self.dataset_dir, self.im_dir, self.mat_file)
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
        if not osp.exists(self.im_dir):
            raise RuntimeError("'{}' is not available".format(self.im_dir))
        if not osp.exists(self.mat_file):
            raise RuntimeError("'{}' is not available".format(self.mat_file))

    def _process_dir(self, dir_path, im_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(im_path, '*.jpg')))
        mat = sio.loadmat(osp.join(dir_path,'cars_annos.mat'))['annotations'][0]
        train_dataset = []
        test_dataset = []
        train_pid_container = set()
        test_pid_container = set()
        for i, img_path in enumerate(img_paths):
            info_im = mat[i]
            if osp.join(dir_path,info_im[0][0]) != img_path:
                for info in mat:
                    if osp.join(dir_path,info[0][0]) == img_path:
                        info_im = info
                        break
            # if info_im[6][0][0] == 1:
            #     test_dataset.append((img_path, info_im[5][0][0], -1))
            #     test_pid_container.add(info_im[5][0][0])
            #     continue
            if info_im[5][0][0] > 98:
                test_dataset.append((img_path, info_im[5][0][0],info_im[1][0][0],info_im[2][0][0],info_im[3][0][0],info_im[4][0][0]))
                test_pid_container.add(info_im[5][0][0])
                continue
            train_pid_container.add(info_im[5][0][0])
            train_dataset.append((img_path, info_im[5][0][0]-1,info_im[1][0][0],info_im[2][0][0],info_im[3][0][0],info_im[4][0][0]))

        num_train_pids = len(train_pid_container)
        num_train_imgs = len(train_dataset)

        num_test_pids = len(test_pid_container)
        num_test_imgs = len(test_dataset)
        return (train_dataset, num_train_pids, num_train_imgs), (test_dataset, num_test_pids, num_test_imgs)

    def get_testSet(self):
        return self.test, self.num_test_pids
