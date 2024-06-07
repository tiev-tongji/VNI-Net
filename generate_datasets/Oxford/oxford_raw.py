# Functions and classes operating on a raw Oxford dataset

import numpy as np
import os
import torch
from typing import List
from torch.utils.data import Dataset
import random

from misc.point_clouds import PointCloudLoader

from torch.utils.data import Dataset, ConcatDataset
from sklearn.neighbors import KDTree

# from datasets.Oxford.utils import *



class OxfordPointCloudLoader(PointCloudLoader):
    def __init__(self,n_points):
        super().__init__()
        self.n_points = n_points
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -1.5

    def read_pc(self, file_pathname: str):
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
       
        pc = np.fromfile(file_pathname, dtype=np.float64)

        # PC in Mulran and Oxford is of size [num_points, 4] -> x,y,z,reflectance
        pc = np.reshape(pc, (-1, 3))
        assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(file_pathname)
        return pc


class OxfordSequence(Dataset):
    """
    Point cloud from a sequence from a raw Mulran dataset
    """
    def __init__(self, dataset_root: str, sequence_name: str, split: str,
                 remove_zero_points: bool = True):
        # pose_time_tolerance: (in seconds) skip point clouds without corresponding pose information (based on
        #                      timestamps difference)/media/autolab/disk_3T/222nas/Datasets/Oxford/odometry/sequences/00/
        # remove_zero_points: remove (0,0,0) points

        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        # self.sequence_path = os.path.join(self.dataset_root, 'sequences')
        # assert os.path.exists(self.sequence_path), f'Cannot access sequence: {self.sequence_path}'
        self.rel_lidar_path = os.path.join(self.sequence_name, 'vel_random_sample_4096')
        # lidar_path = os.path.join(self.sequence_path, self.rel_lidar_path)
        # assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'
        self.pose_file = os.path.join(self.dataset_root, self.sequence_name,'pose.txt')
        assert os.path.exists(self.pose_file), f'Cannot access sequence pose file: {self.pose_file}'
        self.times_file = os.path.join(self.dataset_root, self.sequence_name, 'times.txt')
        assert os.path.exists(self.times_file), f'Cannot access sequence times file: {self.times_file}'
        self.calib_file = os.path.join(self.dataset_root, self.sequence_name, 'calib.txt')
        assert os.path.exists(self.calib_file), f'Cannot access sequence times file: {self.calib_file}'
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.remove_zero_points = remove_zero_points

        self.timestamps, self.poses, filenames = self._read_lidar_poses()
        self.rel_scan_filepath = [os.path.join(self.rel_lidar_path, '%06d%s' % (e, '.bin')) for e in filenames]

        self.kdtree = KDTree(self.get_xy())

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, ndx):
        scan_filepath = os.path.join(self.dataset_root, self.rel_scan_filepath[ndx])
        pc = load_pc(scan_filepath)
        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]
        return {'pc': pc, 'pose': self.lidar_poses[ndx], 'ts': self.rel_lidar_timestamps[ndx]}

    def get_xy(self):
        # Get X, Y position from (4, 4) pose
        return self.poses[:, [0,2],3]

    def _read_lidar_poses(self):
        fnames = os.listdir(os.path.join(self.dataset_root, self.rel_lidar_path))
        temp = os.path.join(self.dataset_root, self.rel_lidar_path)
        fnames = [e for e in fnames if os.path.isfile(os.path.join(temp, e))]
        assert len(fnames) > 0, f"Make sure that the path {self.rel_lidar_path}"
        filenames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        with open(self.pose_file, "r") as h:
            txt_poses = h.readlines()

        n = len(txt_poses)
        poses = np.zeros((n, 4, 4), dtype=np.float64)  # 4x4 pose matrix

        for ndx, pose in enumerate(txt_poses):
            # Split by comma and remove whitespaces
            temp = [e.strip() for e in pose.split(' ')]
            assert len(temp) == 12, f'Invalid line in global poses file: {temp}'
            # poses in Oxford ar ein cam0 reference
            poses[ndx] = np.array([[float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])],
                                   [float(temp[4]), float(temp[5]), float(temp[6]), float(temp[7])],
                                   [float(temp[8]), float(temp[9]), float(temp[10]), float(temp[11])],
                                   [0., 0., 0., 1.]])
        rel_ts = np.genfromtxt(self.times_file)

        return rel_ts, poses, filenames


    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)    
        

            

        


class OxfordSequences(Dataset):
    """
    Multiple Oxford sequences indexed as a single dataset. Each element is identified by a unique global index.
    """
    def __init__(self, dataset_root: str, sequence_names: List[str], split: str):
        assert len(sequence_names) > 0
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_names = sequence_names
        self.split = split

        sequences = []
        for seq_name in self.sequence_names:
            ds = OxfordSequence(self.dataset_root, seq_name, split=split)
            sequences.append(ds)
        
        self.dataset=sequences

        # self.dataset = ConcatDataset(sequences)

        # # Concatenate positions from all sequences
        # self.poses = np.zeros((len(self.dataset), 4, 4), dtype=np.float64)
        # self.timestamps = np.zeros((len(self.dataset),), dtype=np.int64)
        # self.rel_scan_filepath = []

        # for cum_size, ds in zip(self.dataset.cumulative_sizes, sequences):# sumulative : [sizeof(dataset1),sizeof(dataset2)]
        #     # Consolidated lidar positions, timestamps and relative filepaths
        #     self.poses[cum_size - len(ds): cum_size, :] = ds.poses
        #     self.timestamps[cum_size - len(ds): cum_size] = ds.timestamps
        #     self.rel_scan_filepath.extend(ds.rel_scan_filepath)

        # assert len(self.timestamps) == len(self.poses)
        # assert len(self.timestamps) == len(self.rel_scan_filepath)

        # Build a kdtree based on X, Y position
        # self.kdtree = KDTree(self.get_xy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]

    # def get_xy(self):
    #     # Get X, Y position from (4, 4) pose
    #     return self.poses[:, [0,2],3]



def load_pc(filepath):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix
    pc = np.fromfile(filepath, dtype=np.float32)
    # PC in Oxford is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))[:, :3]
    return pc
