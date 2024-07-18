from mvae.utils import *


frames = 2500
pose_vae_path = 'runs/0717_223513_690c_local/posevae_c1_e6_l32.pt'
pose0 = load_pose0(pose_vae_path).float().cuda()
model = torch.load(pose_vae_path).cuda()
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
root_anvel = poses[:, 3:6]
root_rot = anvel_to_rotmat(root_anvel)
zero = np.zeros((frames, 1))
root_pos = np.zeros((frames, 3))
for i in range(frames-1):
    vel = root_rot[i] @ poses[i, :3]
    root_pos[i+1] = root_pos[i] + vel
root_pos[:, 2] = 0

jpos = poses[:, 3:69].reshape(-1, 22, 3)
jpos = np.einsum('ijk,ilk->ilj', root_rot, jpos)  # rotate xz along y axis
jpos += root_pos[:, None, :]
root_rot = rot2aa_numpy(root_rot)

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

rbs = RigidBodies(jpos, jori, length=0.1)
jori = rot2aa_numpy(jori).reshape(-1, 22 * 3)
smpl_layer = SMPLLayer(model_type="smplh")
seq = SMPLSequence(
    poses_body=jori[:, 3:],
    poses_root=root_rot,
    smpl_layer=smpl_layer,
    trans=root_pos,
)

v = Viewer()
zero = np.ones((root_pos.shape[0], 1)) * 3
cam_pos = root_pos + np.array([2, 2, 2])
cam = PinholeCamera(cam_pos, root_pos, v.window_size[0], v.window_size[1], viewer=v)
v.run_animations = True
v.scene.add(cam, seq, rbs)
v.set_temp_camera(cam)
v.run()
