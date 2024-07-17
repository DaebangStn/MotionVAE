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

    jori = local_to_global(torch.from_numpy(poses), skel[:, 0], output_format='rotmat')
    jori = to_numpy(jori.reshape(num_frame, -1, 3, 3))

    root_poses = jori[:, 0]
    root_poses_inv = invert_rotmat(root_poses)
    root_face_anvel = face_anvel_from_root_rotmat(root_poses)

    # Truncate position values to make the size same to the velocity
    face_jori = np.matmul(root_poses_inv[:, np.newaxis, :, :], jori)[2:, :, :, :2]
    face_jpos_local = np.einsum('ijk,ilk->ilj', root_poses_inv, jpos_local)[2:]
    face_jvel = np.einsum('ijk,ilk->ilj', root_poses_inv[2:], jvel)
    face_root_vel = np.einsum('ijk,ik->ij', root_poses_inv[2:], root_vel)

    num_frame = num_frame - 2  # 2 is due to the diff operation

    #  267 = 2 + 1 + 3 * 22 + 3 * 22 + 6 * 22
    data = np.concatenate([face_root_vel[:, :2], root_face_anvel[:, None], face_jpos_local.reshape(num_frame, -1),
                           face_jvel.reshape(num_frame, -1), face_jori.reshape(num_frame, -1)], axis=1)
    end_indices = np.array([num_frame - 1])
    out = {
        'data': data,
        'end_indices': end_indices
    }
    np.savez("./res/mocap/mvae1.npz", **out)
