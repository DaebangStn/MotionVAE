import numpy as np

from mvae.utils import *


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

    if output_format == "aa":
        local_oris = rot2aa_numpy(local_oris.reshape((-1, 3, 3)))
        res = local_oris.reshape((-1, n_joints * 3))
    else:
        res = local_oris.reshape((-1, n_joints * 9))
    return res

frames = 500
fps = 30
pose0 = torch.from_numpy(np.load('res/mocap/pose1.npy')).float().cuda()
model = torch.load('runs/0717_160516_0d04/posevae_c1_e6_l32.pt').cuda()
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
root_pos = np.concatenate([root_pos, zero.reshape(-1, 1)], axis=-1)

jpos = poses[:, 3:69].reshape(-1, 22, 3)
jpos_xz = np.einsum('ijk,ilk->ilj', root_rot, jpos[:, :, [0, 2]])  # rotate xz along y axis
jpos = np.concatenate([jpos_xz, jpos[:, :, 1:2]], axis=-1)  # y-up
jpos += root_pos[:, None, :]

jori_6d = poses[:, -132:].reshape(-1, 22, 3, 2)
jori = np.zeros((frames, 22, 3, 3))

# 6d orientation to 3d rotation matrix
for f in range(frames):
    for j in range(22):
        v1 = jori_6d[f, j, :, 0]
        v2 = jori_6d[f, j, :, 1]

        v1 /= np.linalg.norm(v1)
        v2 -= np.dot(v1, v2) * v1
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        jori[f, j] = np.stack([v1, v2, v3], axis=-1)

smpl_layer = SMPLLayer(model_type="smplh")
parents = smpl_layer.skeletons()["body"].T[:, 0]
jori = global_to_local(jori.reshape(-1, 22*9), parents, output_format="rotmat", input_format="rotmat")
jori = jori.reshape(-1, 22, 3, 3)
rbs = RigidBodies(jpos, jori, length=0.1)

col1 = np.stack([np.cos(yaw), np.sin(yaw), zero], axis=-1)
col2 = np.stack([-np.sin(yaw), np.cos(yaw), zero], axis=-1)
col3 = np.stack([zero, zero, one], axis=-1)
root_rot = np.stack([col1, col2, col3], axis=-1)
# jori = np.einsum('ijk,ilkm->iljm', root_rot, jori)
jori = rot2aa_numpy(jori).reshape(-1, 22*3)
root_rot = rot2aa_numpy(root_rot)

seq = SMPLSequence(
    poses_body=jori[:, 3:],
    poses_root=jori[:, :3],
    smpl_layer=smpl_layer,
    trans=root_pos,
)

v = Viewer()
zero = np.ones((root_pos.shape[0], 1)) * 3
cam_pos = root_pos + np.array([5, 5, 5])
cam = PinholeCamera(cam_pos, root_pos, v.window_size[0], v.window_size[1], viewer=v)
v.run_animations = True
v.scene.add(cam, rbs, seq)
# v.scene.add(cam, seq)
v.set_temp_camera(cam)
v.run()
