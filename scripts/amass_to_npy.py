from mvae.utils import *
from mvae.utils.args import build_amass_to_npy_arg


if __name__ == "__main__":
    arg = build_amass_to_npy_arg()
    amass = np.load(arg.amass)

    sf = 0
    ef = amass['poses'].shape[0]
    num_frame = ef - sf
    poses = amass['poses'][sf:ef]
    trans = amass['trans'][sf:ef]
    bm = SMPLLayer(model_type="smplh", gender=amass['gender'].tolist())

    if arg.fps:
        fps_in = amass['mocap_framerate']
        if arg.fps != fps_in:
            poses = resample_rotations(poses, fps_in, arg.fps)
            trans = resample_positions(trans, fps_in, arg.fps)

    ROOT_ROT = 1
    NUM_BODY_JOINTS = 21

    poses = poses[:, :(ROOT_ROT + NUM_BODY_JOINTS) * 3]
    seq = SMPLSequence(
        poses_root=poses[:, :ROOT_ROT * 3],
        poses_body=poses[:, ROOT_ROT * 3:],
        smpl_layer=bm,
        betas=amass['betas'],
        trans=trans,
    )

    _, jpos, _, skel = seq.fk()
    root_pos = jpos[:, 0] + trans
    root_vel = estimate_velocity(root_pos, 1)
    jpos_local = jpos - jpos[:, 0:1]

    jvel = estimate_velocity(jpos, 1)

    jori_local = torch.from_numpy(poses)
    jori = local_to_global(jori_local, skel[:, 0], output_format='rotmat')
    jori = to_numpy(jori.reshape(num_frame, -1, 3, 3))

    root_poses = jori[:, 0]
    root_poses_inv = invert_rotmat(root_poses)
    root_andiff = estimate_angle_diff(root_poses, 1)

    # 1. Truncate position values to make the size same to the velocity
    # 2. Remove root joint
    face_jori = np.matmul(root_poses_inv[:, np.newaxis, :, :], jori)[2:, 1:, :, :2]
    face_jpos_local = np.einsum('ijk,ilk->ilj', root_poses_inv, jpos_local)[2:, 1:]
    face_jvel = np.einsum('ijk,ilk->ilj', root_poses_inv[2:, 1:], jvel)
    face_root_vel = np.einsum('ijk,ik->ij', root_poses_inv[2:], root_vel)

    num_frame = num_frame - 2  # 2 is due to the diff operation

    #  265 = 3 + 3 + 3 * 21 + 3 * 21 + 6 * 21
    data = np.concatenate([face_root_vel, root_andiff, face_jpos_local.reshape(num_frame, -1),
                           face_jvel.reshape(num_frame, -1), face_jori.reshape(num_frame, -1)], axis=1)
    end_indices = np.array([num_frame - 1])
    out = {
        'data': data,
        'end_indices': end_indices
    }

    dirpath = osp.dirname(arg.amass)
    filename = osp.basename(arg.amass).replace('amass', 'mvae')

    np.savez(osp.join(dirpath, filename), **out)

    """
        Reconstruct the sequence
    """

    if arg.view:
        root_rot = andiff_to_rotmat(root_andiff) @ root_poses[0]
        root_pos = np.zeros((num_frame, 3))
        for i in range(num_frame - 1):
            vel = root_rot[i] @ data[i, :3]
            root_pos[i + 1] = root_pos[i] + vel

        jpos = data[:, 6:69].reshape(-1, 21, 3)
        jpos = np.einsum('ijk,ilk->ilj', root_rot, jpos)  # rotate xz along y axis
        jpos += root_pos[:, None, :]
        jpos = np.concatenate([root_pos[:, None, :], jpos], axis=1)

        jori_6d = data[:, -126:].reshape(-1, 21, 3, 2)
        jori_6d = roma.special_gramschmidt(torch.from_numpy(jori_6d)).numpy()
        jori2 = np.tile(np.eye(3), (num_frame, 1, 1, 1))
        jori2 = np.concatenate([jori2, jori_6d], axis=1)

        jori2_rot = np.einsum('ijk,ilkm->iljm', root_rot, jori2)
        jori2_local = global_to_local(jori2_rot.reshape(-1, 9*22), skel[:, 0], output_format='rotmat', input_format='rotmat')
        jori2 = jori2_local.reshape(-1, 22, 3, 3)
        root_rot = rot2aa_numpy(root_rot)

        rbs = RigidBodies(jpos, jori2_rot, length=0.1)
        jori2 = rot2aa_numpy(jori2).reshape(-1, 66)
        smpl_layer = SMPLLayer(model_type="smplh")
        seq = SMPLSequence(
            poses_body=jori2[:, 3:],
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
