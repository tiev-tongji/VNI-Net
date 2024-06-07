import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.point_vnn.point_sample_util import *

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    # idx = pairwise_distance.argsort(descending=True)
    # idx = idx[:,:,0:k]
    # pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    # (batch_size, num_points, k)
    return idx,-pairwise_distance

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx,_ = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx,_ = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points)
    pairwise_dist = None
    if idx is None:
        idx,pairwise_dist = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)

    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature,pairwise_dist


def get_graph_feature_cross_and_center(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points)
    if idx is None:
        idx,pairwise_dist = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)

    centroid_xyz = feature.mean(dim=2,keepdim=True).repeat(1, 1, k, 1, 1)


    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x,feature - centroid_xyz, cross, x,x-centroid_xyz), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature,pairwise_dist


def get_graph_feature_cross_and_center_sample(x,sampled_x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    sample_num = sampled_x.size(1)

    x = x.reshape(batch_size,num_points,-1)


    if idx is None:
        idx = knn_point(k,x,sampled_x)

    feature = index_points(x,idx) # [B, npoint, knn, C]

 
    _,  _ ,num_dims= x.size()
    num_dims = num_dims // 3

    # x = x.transpose(2, 1).contiguous()
    # feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, sample_num, k, num_dims, 3)

    centroid_xyz = feature.mean(dim=2,keepdim=True).repeat(1, 1, k, 1, 1)


    sampled_x = sampled_x.view(batch_size, sample_num, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, sampled_x, dim=-1)
    
    feature = torch.cat((feature-sampled_x,feature - centroid_xyz, cross, sampled_x,sampled_x-centroid_xyz), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature,_

def get_graph_double_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx,pairwise_dist = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)

    centroid_xyz = feature.mean(dim=2,keepdim=True)
    x = x.view(batch_size, num_points, 1, num_dims, 3)

    cross_g = torch.cross(feature, x, dim=-1)

    feature_g = torch.cat((x,centroid_xyz,x-centroid_xyz,cross_g), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    centroid_xyz = centroid_xyz.repeat(1, 1, k, 1, 1)
    x = x.repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    
    feature_l = torch.cat((feature-x,feature - centroid_xyz,cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

  
    return feature_g,feature_l,pairwise_dist


def pairwise_distance_mask(pc, k=20):
    # pc: BxNx3
    # k: 20 default
    #pc = tf.truncated_normal([18, 1024, 3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    pc_transpose = pc.transpose(2,1)
    pc_inner = torch.matmul(pc, pc_transpose)
    pc_inner = -2 * pc_inner
    pc_square = torch.sum(torch.square(pc), dim=-1, keepdims=True)
    pc_square_transpose = pc_square.transpose(2,1)
    pairwise_distance = - ( pc_square + pc_inner + pc_square_transpose )

    #b = tf.nn.top_k(a, k=2, sorted=False) # select top k elements along the last dimension
    #b = tf.nn.top_k(a, k=2) # select top k elements along the last dimension
    # b = a.top_k(a, k=20) # select top k elements along the last dimension
    b = pairwise_distance.topk(k=k, dim=-1)[0]
    #kth = tf.reduce_min(b.values, 1, keepdims=True)
    kth = b.min(2, keepdims=True)[0]
    topk = torch.ge(pairwise_distance, kth)
    topk = topk.type(torch.float32)
    return topk

########################################
# input point_xyz[B,1,N,3]
# output region_matrix[B,N,2N]
########################################
def get_region_matrix(input_xyz,pairwise_dist):

    input_xyz = input_xyz.squeeze(1).transpose(-1,-2)
    
    point_xyz_norm = torch.norm(input_xyz,dim=-1,keepdim=True) #[B,N,1]
    point_xyz_norm_transpose = point_xyz_norm.transpose(-1,-2)  #[B,1,N]
    point_xyz_inner = torch.matmul(input_xyz, input_xyz.transpose(2, 1)) #[B,N,N]
    point_xyz_angle = point_xyz_inner / (point_xyz_norm*point_xyz_norm_transpose+0.00001)

    region_matrix = torch.concat([pairwise_dist,point_xyz_angle],axis=-1)#[B,N,2N]
    return region_matrix


def xyz2rotinv(downsampled_xyz, grouped_xyz):
    # downsampled_xyz: [B, M, 3] 采样点 
    # grouped_xyz: [B, M, nsample, 3] M个采样点的n近邻

    batch_size = downsampled_xyz.get_shape()[0].value
    point_num = downsampled_xyz.get_shape()[1].value

    # calculate neighbor centroid
    #centroid_xyz = tf.reduce_mean(grouped_xyz, axis=2)  # [B, N, 3] 质心
    centroid_xyz = get_robust_centroid(grouped_xyz)  # [B, M, 3] 几何中心 在pointnet_util.py

    # calculate intersection point 计算交点
    reference_vector_norm = tf.norm(new_xyz, axis=-1, keepdims=True)  # [B, N, 1]
    reference_vector_unit = new_xyz / (reference_vector_norm + 0.0001)  # [B, N, 3]
    inter_xyz = radius * reference_vector_unit + new_xyz

    # prepare features of center point
    centroid_reference_vector = new_xyz - centroid_xyz
    centroid_reference_dist = tf.norm(centroid_reference_vector, axis=-1, keepdims=True)  # [B, N, 1]

    centroid_inter_vector = inter_xyz - centroid_xyz
    centroid_inter_dist = tf.norm(centroid_inter_vector, axis=-1, keepdims=True)  # [B, N, 1]

    dot_product = tf.reduce_sum(tf.multiply(centroid_reference_vector, centroid_inter_vector), axis=-1,
                                keepdims=True)
    reference_centroid_inter_angle = dot_product / (centroid_reference_dist * centroid_inter_dist + 0.000001)

    inter_reference_vector = new_xyz - inter_xyz
    inter_centroid_vector = centroid_xyz - inter_xyz
    dot_product = tf.reduce_sum(tf.multiply(inter_reference_vector, inter_centroid_vector), axis=-1,
                                keepdims=True)
    reference_inter_centroid_angle = dot_product / (radius * centroid_inter_dist + 0.000001)

    center_point_features = tf.concat([reference_vector_norm, centroid_reference_dist, centroid_inter_dist,
                                       reference_centroid_inter_angle, reference_inter_centroid_angle],
                                      axis=-1)  # [B, N, 5]
    #center_point_features = tf.concat([reference_vector_norm, centroid_reference_dist, centroid_inter_dist],
    #                                  axis=-1)  # [B, N, 5]
    center_point_features_tile = tf.tile(tf.expand_dims(center_point_features, axis=2),
                                         [1, 1, nsample, 1])  # [B, N, K, 5]

    # prepare features of neighbor points
    centroid_xyz_tile = tf.tile(tf.expand_dims(centroid_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_centroid_vector = centroid_xyz_tile - grouped_xyz
    reference_vector_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_reference_vector = reference_vector_tile - grouped_xyz
    inter_pts = tf.tile(tf.expand_dims(inter_xyz, axis=2), [1, 1, nsample, 1])
    neighbor_inter_vector = inter_pts - grouped_xyz

    neighbor_centroid_dist = tf.norm(neighbor_centroid_vector, axis=-1, keepdims=True)
    neighbor_reference_dist = tf.norm(neighbor_reference_vector, axis=-1, keepdims=True)
    neighbor_inter_dist = tf.norm(neighbor_inter_vector, axis=-1, keepdims=True)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_centroid_vector, neighbor_reference_vector), axis=-1,
                                keepdims=True)
    centroid_neighbor_reference_angle = dot_product / (neighbor_centroid_dist *
                                                       neighbor_reference_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_reference_vector, neighbor_inter_vector), axis=-1,
                                keepdims=True)
    reference_neighbor_inter_angle = dot_product / (neighbor_reference_dist *
                                                    neighbor_inter_dist + 0.000001)

    dot_product = tf.reduce_sum(tf.multiply(neighbor_inter_vector, neighbor_centroid_vector), axis=-1,
                                keepdims=True)
    inter_neighbor_centroid_angle = dot_product / (neighbor_inter_dist *
                                                   neighbor_centroid_dist + 0.000001)

    #################### calculate angle ####################
    reference_plane_params = get_plane_equation(inter_pts, reference_vector_tile, centroid_xyz_tile)  # [B, N, K, 4]
    reference_normal_vector = reference_plane_params[:, :, :, 0:3]
    # reference_normal_length = tf.norm(reference_normal_vector, axis=-1, keepdims=True)  # [B, N, K, 1]

    neighbor_plane_params = get_plane_equation(inter_pts, reference_vector_tile, grouped_xyz)
    neighbor_normal_vector = neighbor_plane_params[:, :, :, 0:3]
    # neighbor_normal_length = tf.norm(neighbor_normal_vector, axis=-1, keepdims=True)  # [B, N, K, 1]

    dot_product = tf.reduce_sum(tf.multiply(reference_normal_vector, neighbor_normal_vector), axis=-1,
                                keepdims=True)
    # cos_plane_angle = dot_product / (reference_normal_length * neighbor_normal_length + 0.000001)
    cos_plane_angle = dot_product
    plane_angle = tf.acos(cos_plane_angle)  # [B, N, K, 1]  in [0, pi]

    pos_state = tf.reduce_sum(tf.multiply(reference_normal_vector, -neighbor_reference_vector), axis=-1,
                                keepdims=True)
    pos_state = tf.sign(pos_state)   # [B, N, K, 1]
    plane_angle_direction = plane_angle * pos_state  # [0, pi)
    # # sin_plane_angle = tf.sin(plane_angle_direction)
    #
    angle = tf.cos(0.25*plane_angle_direction) - tf.sin(0.25*plane_angle_direction) - 0.75
    # angle = plane_angle_direction/math.pi
    # angle = tf.sin(plane_angle_direction/2.0)
    ###############################################################

    neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist,
                                         centroid_neighbor_reference_angle, reference_neighbor_inter_angle,
                                         inter_neighbor_centroid_angle, angle], axis=-1)  # [B, N, K, 6]
    #neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist, angle], axis=-1)  # [B, N, K, 6]
    #neighbor_point_features = tf.concat([neighbor_centroid_dist, neighbor_reference_dist, neighbor_inter_dist,
    #                                     centroid_neighbor_reference_angle, reference_neighbor_inter_angle,
    #                                     inter_neighbor_centroid_angle], axis=-1)  # [B, N, K, 6]

    rotinv_features = tf.concat([center_point_features_tile, neighbor_point_features], axis=-1)

    # rotinv_features = neighbor_point_features

    return center_point_features_tile, neighbor_point_features, rotinv_features

