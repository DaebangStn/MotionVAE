from mvae.utils import *


frames = 500
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
