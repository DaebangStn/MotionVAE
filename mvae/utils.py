import copy
import os
import os.path as osp
from typing import Optional, Tuple, List
from datetime import datetime
import time
import multiprocessing as mp
from types import SimpleNamespace

import gym
import argparse
from argparse import ArgumentParser
import numpy as np
from git import Repo
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.utils.so3 import resample_rotations, interpolate_rotations, aa2rot_numpy, rot2aa_numpy
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
