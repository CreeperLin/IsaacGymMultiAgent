from typing import List, Optional, Union
from gym.vector.vector_env import VectorEnv as GymVectorEnv
from gymnasium.vector.vector_env import VectorEnv as GymnasiumVectorEnv
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from isaacgymenvs.tasks.base.vec_task import VecTask
from .transformed import TransformedVecTask


class IGMAVectorEnvCommon():

    def __init__(self, env: VecTask, **kwargs):
        t_env = TransformedVecTask(env, **kwargs)
        super().__init__(env.num_environments, t_env.observation_space, t_env.action_space)
        self.t_env = t_env
        self.env = env
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self, **kwargs):
        return self.t_env.step(self.actions)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        infos = {}
        return self.t_env.reset(), infos


class IGMAGymVectorEnv(IGMAVectorEnvCommon, GymVectorEnv):
    pass


class IGMAGymnasiumVectorEnv(IGMAVectorEnvCommon, GymnasiumVectorEnv):

    def step_wait(self, **kwargs):
        return convert_to_terminated_truncated_step_api(self.t_env.step(self.actions), True)
