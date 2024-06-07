
import numpy as np
import argparse
import tqdm
import pickle
import os
import sys
import random
import open3d as o3d 

n_point = 4096

def bin2pcd(pcd_path,points):
    pcd_file = open(pcd_path, 'w', -1)
    pcd_file.write("VERSION 0.7\n" + "FIELDS x y z\n" +
                   "SIZE 4 4 4\n" + "TYPE F F F\n" + "COUNT 1 1 1\n" + "WIDTH " + str(points.shape[0]) +
                   "\nHEIGHT 1\n" + "VIEWPOINT 0 0 0 1 0 0 0\n" + "POINTS " + str(points.shape[0]) + "\nDATA ascii\n\n")
    for point in points:
        pcd_file.write(str(point[0]) + ' ' +
                       str(point[1]) + ' ' + str(point[2])  + '\n')
        pass
    pcd_file.close()

def farthest_point_sample_tensor(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # 初始化一个centroids矩阵，用于存储npoint个采样点的索引位置，大小为B×npoint
    # 其中B为BatchSize的个数
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance矩阵(B×N)记录某个batch中所有点到某一个点的距离，初始化的值很大，后面会迭代更新
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices初始化为0~(B-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 直到采样点达到npoint，否则进行如下迭代：
    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest
        # 取出该中心点centroid的坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
        # 随着迭代的继续，distance矩阵中的值会慢慢变小，
        # 其相当于记录着某个Batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance#确保拿到的是距离所有已选中心点最大的距离。比如已经是中心的点，其dist始终保持为	 #0，二在它附近的点，也始终保持与这个中心点的距离
        distance[mask] = dist[mask]
        # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
        farthest = torch.max(distance, -1)[1]
    return centroids

def downsample_point_cloud(xyzr, voxel_size=0.05):
    # Make xyz pointcloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzr[:,:3])

    # Downsample point cloud using open3d functions
    pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    inv_ids = [ids[0] for ids in ds_ids]
    ds_reflectances = np.asarray(xyzr[:, 3])[inv_ids]
    return np.hstack((pcd_ds.points, ds_reflectances.reshape(-1,1)))


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10

    for i in range(npoint):
        if i == 0:
            centroid = np.mean(xyz,axis=1,keepdims=True)
        else:
            centroids[i-1] = farthest
            centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def in_cloud_sample(raw_pc,num_points = 4096,vox_sz = 0.3):
    xyzr = np.reshape(raw_pc, (-1, 4))

    r = np.linalg.norm(xyzr[:,:3], axis = 1) #求xyz每个维度的二范数
    r_filter = np.logical_and(r > 0.1, r < 80)
    xyzr = xyzr[r_filter]

    # Cap out reflectance
    ref = xyzr[:,3]
    ref[ref > 1000] = 1000 
    xyzr[:,3] = ref 

    # Cut off points past dist_thresh
    dist = np.linalg.norm(xyzr[:,:3], axis=1)
    xyzr = xyzr[dist <= 25]

    # Slowly increase voxel size until we have less than num_points
    while len(xyzr) > num_points:
      xyzr = downsample_point_cloud(xyzr, vox_sz)
      vox_sz += 0.01
    # Re-sample some points to bring to num_points if under num_points 
    ind = np.arange(xyzr.shape[0])
    if num_points - len(ind) > 0:
        extra_points_ind = np.random.choice(xyzr.shape[0], num_points - len(ind), replace = False)
        ind = np.concatenate([ind, extra_points_ind])
    xyzr = xyzr[ind,:]
    assert len(xyzr) == num_points

    # Regularize xyz to be between -1, 1 in x,y planes 
    xyzr[:,0] = xyzr[:,0] - xyzr[:,0].mean()
    xyzr[:,1] = xyzr[:,1] - xyzr[:,1].mean()

    scale = np.max(abs(xyzr[:,:2]))
    xyzr[:,:3] = xyzr[:,:3] / 1000
    return xyzr 

def read_pc(file_pathname: str,sample_method: str):
    # Reads the point cloud without pre-processing
    # Returns Nx3 tensor

    raw_pc = np.fromfile(file_pathname, dtype=np.float32)
    # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance

    pc = np.reshape(raw_pc, (-1, 4))[:, :3]

    if sample_method == 'random':
        if pc.shape[0] >=4096:
            ind = np.random.choice(pc.shape[0],4096,replace=False)
            pc = pc[ind,:]
        else:
            ind = np.random.choice(pc.shape[0],4096,replace=True)
            pc = pc[ind,:]
    elif sample_method == 'fps':
        assert len(pc)>=n_point,f"the source {file_pathname} point_num is less than {n_point}"
        # bin2pcd('/media/autolab/disk_3T/TGX/pcd/origin.pcd',pc)
        pc = farthest_point_sample(pc,n_point)
        # bin2pcd('/media/autolab/disk_3T/TGX/pcd/sampled.pcd',pc)
    elif sample_method == 'incloud':
        pc = in_cloud_sample(raw_pc)
        return pc[:,:3]
   # coords are within -1..1 range in each dimension
    assert pc.shape[0] == n_point, "Error in point cloud shape: {}".format(file_path)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m

    # pc = torch.tensor(pc, dtype=torch.float)
    return pc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training sets for KItti dataset')
    parser.add_argument('--dataset_root',default='/home/autolab/TGX/225nas/Datasets/KITTI/odometry/sequences',)

    args = parser.parse_args()
    train_sequences = []
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    all_folders = sorted(os.listdir(base_path))
    index_list = range(len(all_folders) - 1)[0:11]
    for index in index_list:
        train_sequences.append(all_folders[index])

    for sequence in train_sequences:
        rel_lidar_path = os.path.join(sequence, 'velodyne_no_ground')
        fnames = os.listdir(os.path.join(base_path, rel_lidar_path))
        temp = os.path.join(base_path, rel_lidar_path)
        print(f'dir: {temp}')
        fnames = [e for e in fnames if os.path.isfile(os.path.join(temp, e))]
        assert len(fnames) > 0, f"Make sure that the path {self.rel_lidar_path}"
        filenames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        rel_scan_filepath = [os.path.join(temp, '%06d%s' % (e, '.bin')) for e in filenames]
        folder_path = os.path.join(base_path,sequence,'vel_random_incloud_sample_4096')
        print(f'start process {folder_path}')
        if not os.path.exists(folder_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder_path)
        for file in tqdm.tqdm(rel_scan_filepath):
            pc = read_pc(file,'incloud')
            outpath =  os.path.join(folder_path,file[86:])
            out =pc.tofile(outpath)
            print(f"finish save {outpath}")