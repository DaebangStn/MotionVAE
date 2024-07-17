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
import yaml
import argparse
from argparse import ArgumentParser
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


PROJECT_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
matplotlib.use('TkAgg')


def get_current_git_hash(reduce: Optional[int] = None) -> str:
    repo = Repo(PROJECT_ROOT_PATH)
    commit_hash = str(repo.head.commit.hexsha)
    if reduce is not None:
        commit_hash = commit_hash[:reduce]
    return commit_hash


def logdir_path() -> str:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # name = '_' + cfg_train.get('name', 'noname')
    git_hash = '_' + get_current_git_hash(4)
    # memo = cfg_train.get('memo', '')
    # memo = f'_{memo}' if memo else ''
    # return f'runs/{timestamp}{name}{git_hash}{memo}'
    return f'runs/{timestamp}{git_hash}'


def load_pose0(pose_vae_path: str) -> torch.Tensor:
    basepath = osp.dirname(pose_vae_path)
    cfg_file = osp.join(basepath, 'cfg.yaml')
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    mocap_file = cfg['mocap']
    data = torch.from_numpy(np.load(mocap_file)['data'])
    pose0_idx = torch.randint(0, data.shape[0], (1,))  # sample initial frame
    return data[pose0_idx:pose0_idx + 1]


