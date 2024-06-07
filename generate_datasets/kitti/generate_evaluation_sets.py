# Test set for Kitti Sequence 00 dataset.
# Following procedures in [cite papers Kitti for place reco] we use 170 seconds of drive from sequence for map generation
# and the rest is left for queries

import numpy as np
import argparse
from typing import List
import os
import sys
sys.path.remove('/media/autolab/disk_3T/TGX/MinkLoc3D')

from datasets.kitti.kitti_raw import KittiSequence
from datasets.base_datasets import EvaluationTuple, EvaluationSet
from datasets.dataset_utils import filter_query_elements


MAP_TIMERANGE = (0, 170)


def get_scans(sequence: KittiSequence, min_displacement: float = 0.1, ts_range: tuple = None) -> List[EvaluationTuple]:
    # Get a list of all point clouds from the sequence (the full sequence or test split only)

    elems = []
    old_pos = None
    count_skipped = 0
    displacements = []

    for ndx in range(len(sequence)):
        if ts_range is not None:
            if (ts_range[0] > sequence.rel_lidar_timestamps[ndx]) or (ts_range[1] < sequence.rel_lidar_timestamps[ndx]):#如果点云时间戳不在ts_range范围内则舍弃
                continue
        pose = sequence.lidar_poses[ndx]
        # Kitti poses are in camera coordinates system where where y is upper axis dim
        position = pose[[0,2], 3]#kitti里面x和z是xy坐标

        if old_pos is not None:
            displacements.append(np.linalg.norm(old_pos - position)) #记录相邻两帧之间的距离，如果距离过小，即车辆移动距离很小，则舍弃这一帧

            if np.linalg.norm(old_pos - position) < min_displacement:
                # Ignore the point cloud if the vehicle didn't move
                count_skipped += 1
                continue

        item = EvaluationTuple(sequence.rel_lidar_timestamps[ndx], sequence.rel_scan_filepath[ndx], position, pose)
        elems.append(item)
        old_pos = position

    print(f'{count_skipped} clouds skipped due to displacement smaller than {min_displacement}')
    print(f'mean displacement {np.mean(np.array(displacements))}')
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, min_displacement: float = 0.1,
                            dist_threshold: float = 5.) -> EvaluationSet:

    sequence = KittiSequence(dataset_root, map_sequence)#读取sequence雷达的点云存放位置、点云pose以及点云时间戳

    map_set = get_scans(sequence, min_displacement, MAP_TIMERANGE) # 生成EvaluationTuple，包括点云时间戳、点云文件位置、点云4*4pose以及点云xy平面Pose(0-1700)构建map
    query_set = get_scans(sequence, min_displacement, (MAP_TIMERANGE[-1], sequence.rel_lidar_timestamps[-1]))#(1700-4100)构建query
    query_set = filter_query_elements(query_set, map_set, dist_threshold)# 找到所有query在map中有五米内正样本的作为最终的query
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set) #返回EvaluationSet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for KItti dataset')
    parser.add_argument('--dataset_root',default='/media/autolab/disk_3T/222nas/zhinengche_data/Datasets/KITTI/odometry/sequences',)
    parser.add_argument('--min_displacement', type=float, default=0.1)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=5.)

    args = parser.parse_args()

    # Sequences are fixed
    sequence = '00'
    print(f'Dataset root: {args.dataset_root}')
    print(f'Kitti sequence: {sequence}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    kitti_eval_set = generate_evaluation_set(args.dataset_root, sequence, min_displacement=args.min_displacement,
                                             dist_threshold=args.dist_threshold)
    file_path_name = os.path.join(args.dataset_root, f'kitti_{sequence}_eval.pickle')
    print(f"Saving evaluation pickle: {file_path_name}")
    kitti_eval_set.save(file_path_name)
