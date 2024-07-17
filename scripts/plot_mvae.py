import matplotlib.pyplot as plt

from mvae.utils import *


frames = 500
fps = 30
pose0 = torch.from_numpy(np.load('res/mocap/pose1.npy')).float().cuda()
model = torch.load('runs/0716_202743_d8ff/posevae_c1_e6_l32.pt').cuda()
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
poses = poses[:, 11:].reshape(-1, 38, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def update(frame_idx: int):
    pos = poses[frame_idx]
    ax.clear()
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='o')


ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)


plt.show()
