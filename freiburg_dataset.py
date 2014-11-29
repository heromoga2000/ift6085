#!/usr/bin/env python
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from glob import glob
import matplotlib.image as mpimg
import os


class freiburg_circle(DenseDesignMatrix):
    def __init__(self,
                 which_set='train',
                 datapath=None,):
        """
        Wrapper for dataset
        http://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop_validation.tgz
        from
        http://vision.in.tum.de/data/datasets/rgbd-dataset
        """
        self.img_shape = (28, 28)
        self.img_size = np.prod(self.img_shape)
        self.which_set = which_set
        if datapath is None:
            self.datapath = os.path.join(os.path.expanduser("~"),
                                         "ift6085_data",
                                         "rgbd_dataset_freiburg3_nostructure_texture_near_withloop_validation",
                                         "rgb")
        else:
            self.datapath = datapath
        files = glob(os.path.join(self.datapath, "*_red.png"))
        files = sorted(files,
                       key=lambda x: float(x.split("/")[-1][:-8]))
        X = np.array([mpimg.imread(f) for f in files])
        train_stop = int(.8 * len(X))
        valid_stop = int(.9 * len(X))
        if which_set == 'train':
            X = X[:train_stop]
        elif which_set == 'valid':
            X = X[train_stop:valid_stop]
        elif which_set == 'test':
            X = X[valid_stop:]
        elif which_set == 'full':
            X = X
        else:
            raise ValueError("Value %s for which_set is not supported!"
                             % which_set)
        X = X.reshape(len(X), -1)
        super(freiburg_circle, self).__init__(y=None,
                                              X=X,)
