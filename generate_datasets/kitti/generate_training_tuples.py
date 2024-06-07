# Training tuples generation for Kitti dataset.

import numpy as np
import argparse
import tqdm
import pickle
import os
import sys
# sys.path.remove('/media/autolab/disk_3T/TGX/MinkLoc3D')

from datasets.kitti.kitti_raw import * 
from datasets.base_datasets import TrainingTuple
from datasets.kitti.utils import *
from misc.point_clouds import icp



DEBUG = False

def load_pc(file_pathname):
    # Load point cloud, clip x, y and z coords (points far away and the ground plane)
    # Returns Nx3 matrix
    pc = np.fromfile(file_pathname, dtype=np.float32)
    # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))[:, :3]

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

def generate_training_tuples(ds: KittiSequences, pos_threshold: float = 10, neg_threshold: float = 50):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    tuples_num = 0
    for sequence_ndx in ds:
        for anchor_ndx in tqdm.tqdm(range(len(sequence_ndx))):
            anchor_pos = sequence_ndx.get_xy()[anchor_ndx]

            # Find timestamps of positive and negative elements
            positives = sequence_ndx.find_neighbours_ndx(anchor_pos, pos_threshold)
            non_negatives = sequence_ndx.find_neighbours_ndx(anchor_pos, neg_threshold)
            # Remove anchor element from positives, but leave it in non_negatives
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
                anchor_pc = load_pc(os.path.join(sequence_ndx.dataset_root, sequence_ndx.rel_scan_filepath[anchor_ndx]))
                anchor_pose = sequence_ndx.poses[anchor_ndx]
                for positive_ndx in positives:
                    positive_pc = load_pc(os.path.join(sequence_ndx.dataset_root, sequence_ndx.rel_scan_filepath[positive_ndx]))
                    positive_pose = sequence_ndx.poses[positive_ndx]
                    # Compute initial relative pose
                    transform = get_relative_pose_LCD(anchor_pose, positive_pose,sequence_ndx.calib_file)# icp初值
                    # Refine the pose using ICP
                    m, fitness, inlier_rmse = icp(anchor_pc, positive_pc, transform)

                    fitness_l.append(fitness)
                    inlier_rmse_l.append(inlier_rmse)
                    positive_poses[positive_ndx] = m

            # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
            tuples[anchor_ndx+tuples_num] = TrainingTuple(id=anchor_ndx+tuples_num, timestamp=sequence_ndx.timestamps[anchor_ndx],
                                            rel_scan_filepath=sequence_ndx.rel_scan_filepath[anchor_ndx],
                                            positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                            positives_poses=positive_poses)
        tuples_num = tuples_num + anchor_ndx
        print(f'finish make tuples on{sequence_ndx.sequence_name}:02d')
        print(f'{len(tuples)} training tuples generated')
        print('ICP pose refimenement stats:')
        print(f'Fitness - min: {np.min(fitness_l):0.3f}   mean: {np.mean(fitness_l):0.3f}   max: {np.max(fitness_l):0.3f}')
        print(f'Inlier RMSE - min: {np.min(inlier_rmse_l):0.3f}   mean: {np.mean(inlier_rmse_l):0.3f}   max: {np.max(inlier_rmse_l):0.3f}')

        pickle_name = f'sequence_{sequence_ndx.sequence_name}_{pos_threshold}_{neg_threshold}.pickle'
        train_tuples_filepath = os.path.join(sequence_ndx.dataset_root,sequence_ndx.sequence_name, pickle_name)
        pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))
        

    return tuples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training sets for KItti dataset')
    parser.add_argument('--dataset_root',default='/home/autolab/TGX/225nas/zhinengche_data/Datasets/KITTI/odometry/sequences',)
    parser.add_argument('--pos_threshold', default=3)
    parser.add_argument('--neg_threshold', default=20)
    parser.add_argument('--pose_time_tolerance', type=float, default=1.)
    args = parser.parse_args()

    train_sequences = []
    test_sequences = '08'
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    all_folders = sorted(os.listdir(base_path))
    index_list = range(len(all_folders) - 1)[0:11]
    for index in index_list:
        train_sequences.append(all_folders[index])

    # train_sequences.remove(test_sequences)


    print(f'Dataset root: {args.dataset_root}')
    print(f'Sequences: {train_sequences}')
    # print(f'Test:{test_sequences}')
    print(f'Threshold for positive examples: {args.pos_threshold}')
    print(f'Threshold for negative examples: {args.neg_threshold}')
    print(f'Minimum displacement between consecutive anchors: {args.pose_time_tolerance}')

    ds = KittiSequences(args.dataset_root, train_sequences, split='train')
    train_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
    print(f'finish generate tuples{train_sequences}')
    # pickle_name = f'train_{train_sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
    # train_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
    # pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))
    # train_tuples = None

    # ds = KittiSequences(args.dataset_root, test_sequences, split='test')
    # test_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
    # pickle_name = f'val_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
    # test_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
    # pickle.dump(test_tuples, open(test_tuples_filepath, 'wb'))