import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size  #输入局部特征维度：1024
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size #聚类个数：64
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_conv =  torch.nn.Linear(feature_size, cluster_size, bias=False)
        self.hidden1_conv = torch.nn.Linear(cluster_size * feature_size, output_dim, bias=False)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        # self.hidden1_weights = nn.Parameter(
            # torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

        print('using global net netvlad')

    def forward(self, x):
        #input:[B,1,N,C]
        x = x.transpose(1, 3).contiguous() #[B,C,N,1]
        x = x.view((-1,1, self.max_samples, self.feature_size))#B*N*C
        activation = self.cluster_conv(x).squeeze(1)
        # activation = torch.matmul(x, self.cluster_weights) #B*N*C @ C*k >>> B*N*k  k:cluster_size
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size) #[B*N,K]
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size) #[B,N,K]
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation) #B*N*k 求出a
        activation = activation.view((-1, self.max_samples, self.cluster_size)) #B*N*k 每一个局部特征xi对聚类中心ck都有一个权重a

        a_sum = activation.sum(-2, keepdim=True)#B*1*k 先把一个聚类中心的对于不同的点的权重加起来，然后逐元素乘到聚类中心上
        a = a_sum * self.cluster_weights2 #[B,1,K] * [1,C,K]--->[B,C,K]*[B,C,K]

        activation = torch.transpose(activation, 2, 1) #B*k*N
        x = x.view((-1, self.max_samples, self.feature_size)) #B*N*C
        vlad = torch.matmul(activation, x) #[B,K,C]
        vlad = torch.transpose(vlad, 2, 1) #[B,C,K]
        vlad = vlad - a  #[B,C,K]

        vlad = F.normalize(vlad, dim=1, p=2)#[B,C,K]
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))#[B,C*K]
        vlad = F.normalize(vlad, dim=1, p=2)#[B,C*K]
        


        #pointnetvlad 在K*D维度的特征后加了一个全连接层，得到了[B,256]维度特征
        #self.hidden1_weights [K*C,256]
        vlad = vlad.unsqueeze(1)
        vlad = self.hidden1_conv(vlad).squeeze(1)
        # vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation