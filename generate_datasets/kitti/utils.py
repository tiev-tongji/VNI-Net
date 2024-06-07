import numpy as np
from pykitti.utils import read_calib_file

def velo2cam():
    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return velo2cam


def get_relative_pose_EGONN(pose_1, pose_2):
    # as seen in https://github.com/chrischoy/FCGF
    M = (velo2cam() @ pose_1.T @ np.linalg.inv(pose_2.T) @ np.linalg.inv(velo2cam())).T
    return M


def get_relative_pose_LCD(pose_1, pose_2,calib_file):
    data = read_calib_file(calib_file)
    cam0_to_velo = np.reshape(data['Tr'], (3, 4))
    cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
    # lidar_pose1 = np.linalg.inv(cam0_to_velo) @ (pose_1 @ cam0_to_velo)
    # lidar_pose2 = np.linalg.inv(cam0_to_velo) @ (pose_2 @ cam0_to_velo)
    # m = np.linalg.inv(lidar_pose2) @ lidar_pose1
    #用下式简化运算
    m=(cam0_to_velo.T @ pose_1.T @ np.linalg.inv(pose_2.T) @ np.linalg.inv(cam0_to_velo.T)).T
    # !!!!!!!!!! Fix for relative pose !!!!!!!!!!!!!
    m[:3, 3] = -m[:3, 3]
    return m  

