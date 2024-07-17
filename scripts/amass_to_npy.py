import matplotlib.pyplot as plt
import numpy as np
import torch

from mvae.utils import *


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--amass",
        type=str,
        required=False,
        default="res/mocap/amass1.npz",
        help="Path to the npz file containing the AMASS data.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=False,
        help="FPS for the output sequence.",
    )
    parser.add_argument(
        "--view",
        action='store_true',
        help="View the animation.",
    )
    arg = parser.parse_args()

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

    jori = torch.from_numpy(poses)
    jori = aa2rot_torch(jori.reshape(-1, 22, 3))
    # jori = local_to_global(jori, skel[:, 0], output_format='rotmat')
    jori = to_numpy(jori.reshape(num_frame, -1, 3, 3))

    root_poses = jori[:, 0]
    root_poses_inv = invert_rotmat(root_poses)
    root_anvel = estimate_angular_velocity(root_poses, 1)

    # 1. Truncate position values to make the size same to the velocity
    # 2. Remove root joint
    # face_jori = np.matmul(root_poses_inv[:, np.newaxis, :, :], jori)[2:, 1:, :, :2]
    face_jori = jori[2:, 1:, :, :2]
    face_jpos_local = np.einsum('ijk,ilk->ilj', root_poses_inv, jpos_local)[2:, 1:]
    face_jvel = np.einsum('ijk,ilk->ilj', root_poses_inv[2:], jvel)[:, 1:]
    face_root_vel = np.einsum('ijk,ik->ij', root_poses_inv[2:], root_vel)

    num_frame = num_frame - 2  # 2 is due to the diff operation

    #  256 = 3 + 3 + 3 * 21 + 3 * 21 + 6 * 21
    data = np.concatenate([face_root_vel, root_anvel, face_jpos_local.reshape(num_frame, -1),
                           face_jvel.reshape(num_frame, -1), face_jori.reshape(num_frame, -1)], axis=1)
    end_indices = np.array([num_frame - 1])
    out = {
        'data': data,
        'end_indices': end_indices
    }
    np.savez("./res/mocap/mvae1_local.npz", **out)

    if arg.view:
        rbs = RigidBodies(face_jpos_local, face_jori, length=0.1)
        v = Viewer()
        v.run_animations = True
        v.scene.camera.position = np.array([10.0, 2.5, 0.0])
        v.scene.add(seq)
        v.run()
