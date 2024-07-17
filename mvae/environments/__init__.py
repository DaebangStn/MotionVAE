import gym
from gym.envs.registration import registry


def register(_id, *args, **kvargs):
    if _id in registry.keys():
        return
    else:
        return gym.envs.registration.register(_id, *args, **kvargs)


register(_id="RandomWalkEnv-v0", entry_point="mvae.environments.mocap_envs:RandomWalkEnv")
register(_id="TargetEnv-v0", entry_point="mvae.environments.mocap_envs:TargetEnv")
register(_id="JoystickEnv-v0", entry_point="mvae.environments.mocap_envs:JoystickEnv")
register(_id="PathFollowEnv-v0", entry_point="mvae.environments.mocap_envs:PathFollowEnv")
register(_id="HumanMazeEnv-v0", entry_point="mvae.environments.mocap_envs:HumanMazeEnv")
