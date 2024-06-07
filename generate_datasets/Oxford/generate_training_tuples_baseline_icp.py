# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import tqdm

from datasets.base_datasets import TrainingTuple
from misc.point_clouds import icp

# Import test set boundaries
from generate_test_sets import P1, P2, P3, P4, check_in_test_set

# Test set boundaries
P = [P1, P2, P3, P4]

RUNS_FOLDER = "oxford/"
FILENAME = "pointcloud_locations_20m_10overlap.csv"
POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"


def load_pc(file_pathname):
    # Load point cloud, clip x, y and z coords (points far away and the ground plane)
    # Returns Nx3 matrix
    file_path = os.path.join(self.dataset_path, filename)
    pc = np.fromfile(file_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))

    mask = np.all(np.isclose(pc, 0.), axis=1)
    pc = pc[~mask]#去除全0点
    mask = pc[:, 0] > -80
    pc = pc[mask]
    mask = pc[:, 0] <= 80

    pc = pc[mask]
    mask = pc[:, 1] > -80
    pc = pc[mask]
    mask = pc[:, 1] <= 80
    pc = pc[mask]

    mask = pc[:, 2] > -0.9
    pc = pc[mask]
    return pc


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)


         # ICP pose refinement
        fitness_l = []
        inlier_rmse_l = []
        positive_poses = {}

        if DEBUG:
            # Use ground truth transform without pose refinement
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                m, fitness, inlier_rmse = relative_pose(anchor_pose, positive_pose), 1., 1.
                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m
        else:
            anchor_pc = load_pc(os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx]))
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pc = load_pc(os.path.join(ds.dataset_root, ds.rel_scan_filepath[positive_ndx]))
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                transform = relative_pose(anchor_pose, positive_pose)# icp初值
                # Refine the pose using ICP
                m, fitness, inlier_rmse = icp(anchor_pc, positive_pc, transform)

                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_scan_filepath=ds.rel_scan_filepath[anchor_ndx],
                                           positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                           positives_poses=positive_poses)

        #每一帧点云构造一个trainingtuple 包括点云的序号，时间戳，点云文件路径，正相关点云序号，反相关点云序号 点云位置
        # # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        # queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
        #                                     positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root',default='/media/autolab/disk_3T/222nas/zhinengche_data/Datasets/Oxford_lidar/',)
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders) - 1)
    print("Number of runs: " + str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    print(folders)

    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, FILENAME), sep=',')
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

#根据给出的P1 P2 P3 P4的位置以及在generate_test_sets.py里面设置的范围划分train 和 test
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "training_queries_baseline.pickle", ind_nn_r=10)
    construct_query_dict(df_test, base_path, "test_queries_baseline.pickle", ind_nn_r=10)
