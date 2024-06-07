from __future__ import print_function
from inspect import GEN_CLOSED
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
from utils.vn_layers import *
# from models.point_vnn.vn_pointnet import *
# from models.point_vnn.vn_dgcnn import EQCNN_cls
from utils.vn_dgcnn_util import *
from scipy.spatial.transform import Rotation as R
from models.netvlad import NetVLADLoupe
from models.gem import GeM
from models.RINet import *

import math

class VNI_Net(nn.Module):
    def __init__(self, args,num_points=2500,global_feat=True, feature_transform=False, max_pool=True, output_dim=1024,normal_channel=True):
        super(VNI_Net, self).__init__()
        # self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
        #                               feature_transform=feature_transform, max_pool=max_pool)
        self.args = args
        self.k = self.args.n_knn

        self.invnet = VNN_RI_Feature(args,num_point = num_points,global_feat=True, feature_transform=feature_transform,max_pool=max_pool)
        
        self.mlp = Last_Mlp(input_dim = 64)
        

        if args.globalnet == 'gem':
            self.globnet = GeM(output_dim)
        elif args.globalnet == 'netvlad':
            self.globnet = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)                

    def forward(self, x):
        # B, N,D = x.size()

        x = self.invnet(x)
        x = self.mlp(x)
        
        if self.args.globalnet == 'netvlad':
            x = x.unsqueeze(1)

        x = self.globnet(x)
        return x
