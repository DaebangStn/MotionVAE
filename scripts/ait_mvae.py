import matplotlib.pyplot as plt
import numpy as np

from mvae.utils import *


frames = 500
fps = 30
pose0 = torch.from_numpy(np.load('mvae/environments/pose0.npy')).float().cuda()
model = torch.load('res/models/posevae_c1_e6_l32.pt').cuda()
model.eval()

poses = []
curr_pose = pose0

z = torch.empty((1, 32)).cuda()

with torch.no_grad():
    for i in range(frames):
        z.normal_(0, 1.0)
        curr_pose = model.normalize(curr_pose)
        curr_pose = model.sample(z, curr_pose)
        curr_pose = model.denormalize(curr_pose)
        poses.append(curr_pose.cpu().numpy())

poses = np.concatenate(poses, axis=0)

yaw = np.cumsum(poses[:, 2], axis=0)
one = np.ones_like(yaw)
zero = np.zeros_like(yaw)
col1 = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
col2 = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
root_rot = np.stack([col1, col2], axis=-1)

root_pos = np.zeros((frames, 2))
for i in range(frames-1):
    vel = root_rot[i] @ poses[i, :2]
    root_pos[i+1] = root_pos[i] + vel

jpos = poses[:, 3:69].reshape(-1, 22, 3)
jpos_xz = np.einsum('ijk,ilk->ilj', root_rot, jpos[:, :, [0, 2]])  # rotate xz along y axis
jpos = np.concatenate([jpos_xz, jpos[:, :, 1:2]], axis=-1)  # y-up
jpos[:, :, :2] += root_pos[:, None, :]

jori = poses[:, -132:].reshape(-1, 22, 3, 2)

# 6d orientation to 3d rotation matrix
jori = np.concatenate([jori, np.cross(jori[:, :, :, 0], jori[:, :, :, 1])[:, :, :, None]], axis=-1)

rbs = RigidBodies(jpos, jori, length=0.1)
v = Viewer()
zero = np.ones((root_pos.shape[0], 1)) * 3
cam_tar = np.concatenate([root_pos, zero], axis=-1)
cam_pos = cam_tar + np.array([5, 5, 5])
cam = PinholeCamera(cam_pos, cam_tar, v.window_size[0], v.window_size[1], viewer=v)
v.run_animations = True
v.scene.add(cam, rbs)
v.set_temp_camera(cam)
v.run()
