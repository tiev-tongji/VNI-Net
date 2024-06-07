import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from utils.vn_layers import *
from utils.vn_dgcnn_util import *
from scipy.spatial.transform import Rotation as R
import math
from multiprocessing import Pool
from models.gem import GeM
# from models.point_vnn.point_sample_util import *


class VNN_RI_Feature(nn.Module):
    def __init__(self, args, num_point = 4096,global_feat=True, feature_transform=False,max_pool=True, channel=3):
        super(VNN_RI_Feature, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.max_pool = max_pool
        self.num_point = num_point

        
        self.conv_pos1 = VNLinearLeakyReLU(5, 64, dim=5, negative_slope=0.0)
        self.conv1_1 = VNLinearLeakyReLU(64 , 64, dim=4, negative_slope=0.0)
        self.conv1_2 = VNLinearLeakyReLU(64 , 64, dim=4, negative_slope=0.0)
        self.conv1_3 = VNAttention(64)

        self.conv2_1 = VNLinearLeakyReLU(64 , 64, dim=4, negative_slope=0.0)
        self.conv2_2 = VNLinearLeakyReLU(64 , 64, dim=4, negative_slope=0.0)
        self.conv2_3 = VNAttention(64)

        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        print('using invnet vnn pointnet')

    def forward(self, x):
        B, N,D = x.size()      

#--------------------VNN_PNV-----------------------------------------------------
        x=x.transpose(1,2)
        x = x.unsqueeze(1) #[B,1,3,N]
  
        if self.num_point != 4096:
            feat,pointwise_dist = get_graph_feature_cross_and_center(x,sampled_xyz,k=self.n_knn)
        else:
            feat,pointwise_dist,idx = get_graph_feature_cross_and_center(x, k=self.n_knn)

        x1 = self.conv_pos1(feat)
        x1_1 = self.pool(x1)#[B,N,64,3]

        x1_2 = self.conv1_1(x1_1)+x1_1 #[B,N,64,3]
        x1_3 = self.conv1_2(x1_2)+x1_2 #[B,N,64,3]

        x1_4 = x1_2+x1_3 #[B,N,64,3]

        x1_5 = self.conv1_3(x1_4)+x1_4#[B,N,256,3]

        x1_5 = x1_5.mean(dim=1,keepdim=True)
##########################################################################
        x2_2 = self.conv2_1(x1_4)+x1_4
        x2_3 = self.conv2_1(x2_2)+x2_2

        x2_4 = x2_2+x2_3
        x2_5 = self.conv2_3(x2_4)+x2_4
        x2_5 = x2_5.mean(dim=1,keepdim=True)
##########################################################################
      
        x2 = x1_2 + x2_2 #+ x3_2 + x4_2
        x3 = x1_3 + x2_3 #+ x3_3 + x4_3
        x4 = x1_4 + x2_4 #+ x3_4 + x4_4
        x5 = x1_5 + x2_5 #+ x3_5 + x4_5
        del x1_2,x2_2,x1_3,x2_3,x1_4,x1_5,x2_5,x2_4

##################################### 求欧氏距离 #############################################################################
        x3 = x3.permute(0,3,1,2).reshape(B*N,-1,3)[idx,:]  #[B*N,64,3]
        x3 = x3.reshape(B,N,self.n_knn,-1,3) #[B,N,20,64,3]
        x2 = x2.permute(0,3,1,2).view(B, N, 1, -1, 3).repeat(1, 1, self.n_knn, 1, 1) #[B,N,20,64,3]
        inv_x_cross_ojld = F.pairwise_distance(x2,x3)#[B,N,20,64]
        inv_x_cross_ojld = inv_x_cross_ojld.max(dim=2).values#[B,N,64]

        inv_x_norm = x4.permute(0,3,1,2).norm(dim = -1);

        x5= x5.permute(0,3,2,1).reshape(B*N,-1)[idx,:]  #[B*N,3] 
        x5 = x5.reshape(B,N,self.n_knn,3,1) #[B,N,20,3,1]

        inv_x_cross_cos = torch.einsum('ijkl,ijnlm->ijnkm',x4.permute(0,3,1,2),x5)# 矩阵转置相乘等于不转置直接点积

        inv_x_cross_cos = inv_x_cross_cos.squeeze(-1)   #[B,N,20,64]
        inv_x_cross_cos = inv_x_cross_cos.max(dim=2).values#[B,N,64]

        inv_x = torch.cat([inv_x_cross_ojld,inv_x_cross_cos,inv_x_norm],dim=-1)

        return inv_x
    
class Last_MLP(nn.Module):
    def __init__(self,input_dim = 64//3):
        super(Last_MLP, self).__init__()
        self.input_dim = input_dim
        # self.pnv_conv2 = torch.nn.Linear(self.input_dim, 64, bias=False)
        self.pnv_conv3 = torch.nn.Linear(64, 64, bias=False)
        self.pnv_conv4 = torch.nn.Linear(64, 128, bias=False)
        self.pnv_conv5 = torch.nn.Linear(128, 1024, bias=False)

        self.pnv_bn2 = nn.BatchNorm2d(64)
        self.pnv_bn3 = nn.BatchNorm2d(64)
        self.pnv_bn4 = nn.BatchNorm2d(128)
        self.pnv_bn5 = nn.BatchNorm2d(1024)   
    def forward(self, x):
        if isinstance(x,list):
            x = x[0]
        B, N,D = x.size()
        x = x.unsqueeze(1)
        # x = F.relu(self.pnv_bn2(self.pnv_conv2(x).transpose(1,-1))).transpose(1,-1)
        x = F.relu(self.pnv_bn3(self.pnv_conv3(x).transpose(1,-1))).transpose(1,-1)
        x = F.relu(self.pnv_bn4(self.pnv_conv4(x).transpose(1,-1))).transpose(1,-1)
        x = self.pnv_bn5(self.pnv_conv5(x).transpose(1,-1))

        x = x.view(B,N,-1)
        
        return x
    
    