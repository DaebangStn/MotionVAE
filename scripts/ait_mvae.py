from mvae.utils import *


frames = 500
pose_vae_path = 'runs/0718_163805_6810_global_diff/posevae_c1_e6_l32.pt'
pose0 = load_pose0(pose_vae_path).float().cuda()
model = torch.load(pose_vae_path).cuda()
model.eval()
smpl_layer = SMPLLayer(model_type="smplh")

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
root_rot = andiff_to_rotmat(root_anvel)
zero = np.zeros((frames, 1))
root_pos = np.zeros((frames, 3))
for i in range(frames-1):
    vel = root_rot[i] @ poses[i, :3]
    root_pos[i+1] = root_pos[i] + vel
root_pos[:, 2] = 0

jpos = poses[:, 6:69].reshape(-1, 21, 3)
jpos = np.einsum('ijk,ilk->ilj', root_rot, jpos)  # rotate xz along y axis
jpos += root_pos[:, None, :]
jpos = np.concatenate([root_pos[:, None, :], jpos], axis=1)

jori_6d = poses[:, -126:].reshape(-1, 21, 3, 2)
jori_6d = roma.special_gramschmidt(torch.from_numpy(jori_6d)).numpy()
jori = np.tile(np.eye(3), (frames, 1, 1, 1))
jori = np.concatenate([jori, jori_6d], axis=1)
jori2_rot = np.einsum('ijk,ilkm->iljm', root_rot, jori)

rbs = RigidBodies(jpos, jori2_rot, length=0.1)

jori2_local = global_to_local(jori2_rot.reshape(-1, 9 * 22), smpl_layer.skeletons()["body"].T[:, 0],
                              output_format='rotmat', input_format='rotmat')
jori2 = jori2_local.reshape(-1, 22, 3, 3)
jori2 = rot2aa_numpy(jori2).reshape(-1, 66)
root_rot = rot2aa_numpy(root_rot)
seq = SMPLSequence(
    poses_body=jori2[:, 3:],
    poses_root=jori2[:, :3],
    smpl_layer=smpl_layer,
    trans=root_pos,
)

v = Viewer()
zero = np.ones((root_pos.shape[0], 1)) * 3
cam_pos = root_pos + np.array([2, 2, 2])
cam = PinholeCamera(cam_pos, root_pos, v.window_size[0], v.window_size[1], viewer=v)
v.run_animations = True
v.scene.add(cam, seq)
v.set_temp_camera(cam)
v.run()
