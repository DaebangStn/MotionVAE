from mvae.utils import *


if __name__ == "__main__":
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    seq_amass = SMPLSequence.from_amass(
        npz_data_path='res/mocap/amass1.npz',
        fps_out=60.0,
        color=c,
        name="AMASS Running",
        show_joint_angles=True,
    )

    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(seq_amass)
    v.run()
