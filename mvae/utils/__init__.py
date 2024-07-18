import copy
import os
from glob import glob
import os.path as osp
from typing import Optional, Tuple, List
from datetime import datetime
import time
import multiprocessing as mp
from types import SimpleNamespace

import gym
import roma
import yaml
import argparse
from argparse import ArgumentParser, Namespace
import numpy as np
from git import Repo
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation

from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.utils.so3 import (resample_rotations, interpolate_rotations, aa2rot_numpy, rot2aa_numpy, rot2aa_torch,
                                 aa2rot_torch)
from aitviewer.utils import interpolate_positions, local_to_global, resample_positions, to_numpy, to_torch

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
matplotlib.use('TkAgg')


def get_current_git_hash(reduce: Optional[int] = None) -> str:
    repo = Repo(PROJECT_ROOT_PATH)
    commit_hash = str(repo.head.commit.hexsha)
    if reduce is not None:
        commit_hash = commit_hash[:reduce]
    return commit_hash


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def logdir_path(cfg: dict) -> str:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # name = '_' + cfg_train.get('name', 'noname')
    git_hash = '_' + get_current_git_hash(4)
    memo = cfg.get('memo', '')
    memo = f'_{memo}' if memo else ''
    # return f'runs/{timestamp}{name}{git_hash}{memo}'
    return f'runs/{timestamp}{git_hash}{memo}'


def load_pose0(pose_vae_path: str) -> torch.Tensor:
    basepath = osp.dirname(pose_vae_path)
    cfg_file = osp.join(basepath, 'cfg.yaml')
    mocap_file = load_yaml(cfg_file)['mocap']
    data = torch.from_numpy(np.load(mocap_file)['data'])
    pose0_idx = torch.randint(0, data.shape[0], (1,))  # sample initial frame
    return data[pose0_idx:pose0_idx + 1]


def face_anvel_from_root_rotmat(root_rotmat: np.ndarray) -> np.ndarray:
    assert root_rotmat.ndim == 3, "root_rotmat must be 3D array. (f, 3, 3)"
    forward = root_rotmat[:, :, 0]
    forward[:, 1] = 0  # y-up
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    face_angle = np.arctan2(forward[:, 2], forward[:, 0])
    face_anvel = np.diff(face_angle)
    face_anvel = (face_anvel + np.pi) % (2 * np.pi) - np.pi
    return face_anvel


def estimate_velocity(data_seq, h):
    '''
    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    '''
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2*h)
    return data_vel_seq


def estimate_angle_diff(rot_seq, h):
    '''
    Given a sequence of T rotation matrices, estimates the angle difference at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    Returns the angle difference in axis-angle (T-2, ... , 3) format.
    - h : step size
    '''
    R = rot_seq[2:]
    Rp = rot_seq[1:-1]
    Rdiff = R @ np.swapaxes(Rp, -1, -2)
    return rot2aa_numpy(Rdiff)


def estimate_angular_velocity(rot_seq, h):
    '''
    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w


def invert_rotmat(rotmat: np.ndarray) -> np.ndarray:
    assert rotmat.ndim >= 2, "rotmat must have at least 2 dimensions."

    axes = list(range(rotmat.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]

    # Transpose the array
    inv_rotmat = np.transpose(rotmat, axes=axes)
    return inv_rotmat


def global_to_local(poses: np.ndarray, parents: List[int], output_format="aa", input_format="aa") -> np.ndarray:
    """
    Convert global joint angles to relative ones by unrolling the kinematic chain.
    :param poses: A tensor of shape (N, N_JOINTS*3) defining the global poses in angle-axis format.
    :param parents: A list of parents for each joint j, i.e. parent[j] is the parent of joint j.
    :param output_format: 'aa' or 'rotmat'.
    :param input_format: 'aa' or 'rotmat'
    :return: The global joint angles as a tensor of shape (N, N_JOINTS*DOF).
    """
    assert output_format in ["aa", "rotmat"]
    assert input_format in ["aa", "rotmat"]
    assert poses.ndim == 2
    dof = 3 if input_format == "aa" else 9
    n_joints = poses.shape[-1] // dof
    if input_format == "aa":
        global_oris = aa2rot_numpy(poses.reshape((-1, 3)))
    else:
        global_oris = poses
    global_oris = global_oris.reshape((-1, n_joints, 3, 3))
    local_oris = np.zeros_like(global_oris)

    for j in range(n_joints):
        if parents[j] < 0:
            # root rotation
            local_oris[..., j, :, :] = global_oris[..., j, :, :]
        else:
            parent_rot = global_oris[..., parents[j], :, :]
            global_rot = global_oris[..., j, :, :]
            local_oris[..., j, :, :] = np.matmul(parent_rot.transpose((0, 2, 1)), global_rot)
            # local_oris[..., j, :, :] = np.matmul(global_rot, parent_rot.transpose((0, 2, 1)))

    if output_format == "aa":
        local_oris = rot2aa_numpy(local_oris.reshape((-1, 3, 3)))
        res = local_oris.reshape((-1, n_joints * 3))
    else:
        res = local_oris.reshape((-1, n_joints * 9))
    return res


def anvel_to_rotmat(anvel: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Convert angular velocities to rotation matrices.
    :param anvel: A numpy array of shape (f, 3) representing angular velocities.
    :param dt: Time step between frames (default is 1.0).
    :return: A numpy array of shape (f, 3, 3) of rotation matrices.
    """
    assert isinstance(anvel, np.ndarray)
    assert anvel.shape[-1] == 3
    f = anvel.shape[0]
    rotmat = np.zeros((f, 3, 3))
    rotmat[0] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    for i in range(f - 1):
        w = anvel[i] * dt  # Scale angular velocity by time step
        w_skew = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        delta_R = expm(w_skew)  # Compute incremental rotation
        rotmat[i + 1] = delta_R @ rotmat[i]  # Accumulate rotation
    return rotmat


def andiff_to_rotmat(andiff: np.ndarray) -> np.ndarray:
    """
    Convert angle differences to rotation matrices.
    :param andiff: A numpy array of shape (f, 3) representing angle differences.
    :return: A numpy array of shape (f, 3, 3) of rotation matrices.
    """
    assert isinstance(andiff, np.ndarray)
    assert andiff.shape[-1] == 3
    f = andiff.shape[0]
    Rdiff = aa2rot_numpy(andiff)
    rotmat = np.zeros((f, 3, 3))
    # rotmat[0] = np.eye(3)
    rotmat[0] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    for i in range(f - 1):
        rotmat[i + 1] = Rdiff[i] @ rotmat[i]
    return rotmat
