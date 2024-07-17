from mvae.utils import *


def build_amass_to_npy_arg() -> Namespace:
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
    return arg
