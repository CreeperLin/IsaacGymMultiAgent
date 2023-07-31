import math
import sys
import time
from typing import Dict, Any
from isaacgym import gymapi

_default_sim_params = {
    'dt': 0.01,
    'substeps': 2,
    'up_axis': 'z',
    'use_gpu_pipeline': True,
    'gravity': [0.0, 0.0, -9.81],
    # 'gravity': [0.0, 0.0, 0.0],
    'physx': {
        'num_threads': 0,
        'num_subscenes': 0,
        'solver_type': 1,
        'use_gpu': True,
        'num_position_iterations': 4,
        'num_velocity_iterations': 0,
        'contact_offset': 0.02,
        'rest_offset': 0.001,
        'bounce_threshold_velocity': 0.2,
        'max_depenetration_velocity': 1000.0,
        'default_buffer_size_multiplier': 5.0,
        'max_gpu_contact_pairs': 1048576,
        'contact_collection': 0
    }
}


def create_ground_plane(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)


def default_viewer_cam_setter(gym, sim, viewer, envs):
    env = envs[0]
    cam_pos = [2.0, 2.0, 2.0]
    cam_target = [0.0, 0.0, 0.0]
    # set the camera position based on up axis
    sim_params = gym.get_sim_params(sim)
    if sim_params.up_axis != gymapi.UP_AXIS_Z:
        cam_pos = [cam_pos[0], cam_pos[2], cam_pos[1]]
        cam_target = [cam_target[0], cam_target[2], cam_target[1]]
    gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))


def create_actors(asset_file_setter=None,
                  asset_options_setter=None,
                  asset_setter=None,
                  pose_setter=None,
                  actor_setter=None,
                  env_setter=None,
                  gym=None,
                  sim=None,
                  name=None,
                  asset_root='.',
                  collision_group_setter=None,
                  collision_filter_setter=None,
                  num_envs=1,
                  num_env_actors=1,
                  spacing=1.0):
    num_per_row = int(math.sqrt(num_envs))
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    assets = {}
    envs = []
    actor_handles = []
    for env_idx in range(num_envs):
        env = gym.create_env(sim, lower, upper, num_per_row)
        envs.append(env)
    for env_idx in range(num_envs):
        pose = gymapi.Transform()
        env = envs[env_idx]
        for actor_idx in range(num_env_actors):
            asset_file = asset_file_setter(env_idx, actor_idx)
            asset = assets.get(asset_file)
            if asset is None:
                asset_options = gymapi.AssetOptions()
                if asset_options_setter is not None:
                    asset_options_setter(asset_options)
                asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
                assets[asset_file] = asset
            if asset_setter is not None:
                asset_setter(gym, sim, asset, env_idx, actor_idx)
            if pose_setter is not None:
                pose_setter(pose, env_idx, actor_idx)
            col_group = -1 if collision_group_setter is None else collision_group_setter(env_idx, actor_idx)
            col_filter = -1 if collision_filter_setter is None else collision_filter_setter(env_idx, actor_idx)
            actor_name = '{}_{}_{}'.format(name or 'actor', env_idx, actor_idx)
            actor_handle = gym.create_actor(env, asset, pose, actor_name, col_group, col_filter, 0)
            actor_handles.append(actor_handle)
            if actor_setter is not None:
                actor_setter(gym, sim, env, actor_handle, env_idx, actor_idx)
    if env_setter is not None:
        for env_idx in range(num_envs):
            env_setter(gym, sim, envs[env_idx], env_idx)
    return envs, actor_handles, assets


def init_viewer(gym, sim, events=None):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    for key, event in (events or []):
        if isinstance(key, str):
            key = getattr(gymapi, key)
        gym.subscribe_viewer_keyboard_event(viewer, key, event)
    return viewer


def parse_sim_params(physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    # check correct up-axis
    up_axis = config_sim.get('up_axis', 'z')
    if up_axis not in ['z', 'y']:
        msg = f'Invalid physics up-axis: {up_axis}'
        print(msg)
        raise ValueError(msg)
    # assign general sim parameters
    sim_params.dt = config_sim['dt']
    sim_params.num_client_threads = config_sim.get('num_client_threads', 0)
    sim_params.use_gpu_pipeline = config_sim['use_gpu_pipeline']
    sim_params.substeps = config_sim.get('substeps', 2)
    # assign up-axis
    if up_axis == 'z':
        sim_params.up_axis = gymapi.UP_AXIS_Z
    else:
        sim_params.up_axis = gymapi.UP_AXIS_Y
    # assign gravity
    sim_params.gravity = gymapi.Vec3(*config_sim['gravity'])
    # configure physics parameters
    if physics_engine == 'physx':
        # set the parameters
        if 'physx' in config_sim:
            for opt in config_sim['physx'].keys():
                if opt == 'contact_collection':
                    setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim['physx'][opt]))
                else:
                    setattr(sim_params.physx, opt, config_sim['physx'][opt])
    else:
        # set the parameters
        if 'flex' in config_sim:
            for opt in config_sim['flex'].keys():
                setattr(sim_params.flex, opt, config_sim['flex'][opt])
    # return the configured params
    return sim_params


class Testbench():

    def __init__(
        self,
        asset_file=None,
        asset_file_setter=None,
        asset_root='.',
        asset_setter=None,
        pose_setter=None,
        asset_options_setter=None,
        actor_setter=None,
        env_setter=None,
        scene_setter=create_ground_plane,
        pre_sim_setter=None,
        post_sim_setter=None,
        viewer_setter=default_viewer_cam_setter,
        physics_engine='physx',
        sim_device='cuda:0',
        graphics_device_id=None,
        sim_params=None,
        headless=False,
        events=None,
        event_handlers=None,
        num_envs=1,
        num_env_actors=1,
        collision_group_setter=None,
        collision_filter_setter=None,
        collision_type=None,
        spacing=1.0,
    ):
        split_device = sim_device.split(':')
        device_type = split_device[0]
        device_id = int(split_device[1]) if len(split_device) > 1 else 0
        graphics_device_id = device_id if graphics_device_id is None else graphics_device_id
        device = "cpu"
        sim_params = _default_sim_params.copy() if sim_params is None else sim_params
        if device_type.lower() == "cuda" or device_type.lower() == "gpu":
            device = "cuda" + ":" + str(device_id)
        elif sim_params["use_gpu_pipeline"]:
            print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
            sim_params["use_gpu_pipeline"] = False
        sim_params = parse_sim_params(physics_engine, sim_params)
        if physics_engine == 'physx':
            physics_engine = gymapi.SIM_PHYSX
        elif physics_engine == 'flex':
            physics_engine = gymapi.SIM_FLEX
        else:
            raise ValueError(f'Invalid physics engine backend: {physics_engine}')
        self.device = device
        self.asset_file = asset_file
        self.asset_file_setter = asset_file_setter
        self.asset_root = asset_root
        self.asset_options_setter = asset_options_setter
        self.asset_setter = asset_setter
        self.pose_setter = pose_setter
        self.actor_setter = actor_setter
        self.env_setter = env_setter
        self.scene_setter = scene_setter
        self.pre_sim_setter = pre_sim_setter
        self.post_sim_setter = post_sim_setter
        self.num_envs = num_envs
        self.num_env_actors = num_env_actors
        self.event_handlers = event_handlers or {}
        if collision_type == 'all':

            def collision_group_setter_all(e, a):
                return 0

            collision_group_setter = collision_group_setter_all
        elif collision_type == 'none':
            # def collision_group_setter_none(e, a): return 0
            # def collision_filter_setter_none(e, a): return 1
            def collision_group_setter_none(e, a):
                return e * 0xFFFFF + a

            def collision_filter_setter_none(e, a):
                return 0

            collision_group_setter = collision_group_setter_none
            collision_filter_setter = collision_filter_setter_none
        elif collision_type == 'env':

            def collision_group_setter_env(e, a):
                return e

            collision_group_setter = collision_group_setter_env
        self.collision_group_setter = collision_group_setter
        self.collision_filter_setter = collision_filter_setter
        self.spacing = spacing
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(device_id, graphics_device_id, physics_engine, sim_params)
        assert self.sim is not None
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.headless = headless
        viewer = None
        if not headless:
            viewer = init_viewer(self.gym, self.sim, events=events)
            if viewer_setter is not None:
                viewer_setter(self.gym, self.sim, viewer, self.envs)
        self.viewer = viewer

    def render(self):
        if self.headless:
            return
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            handler = self.event_handlers.get(evt.action, None)
            if handler is not None:
                handler(evt)
        # fetch results
        self.gym.fetch_results(self.sim, True)
        # step graphics
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def step(self):
        if self.pre_sim_setter is not None:
            self.pre_sim_setter(self)
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.post_sim_setter is not None:
            self.post_sim_setter(self)

    def run(self):
        t0 = time.monotonic()
        fps_window = 200
        step = -1
        self.gym.simulate(self.sim)
        while True:
            step += 1
            self.render()
            self.step()
            if (step + 1) == fps_window:
                t1 = time.monotonic()
                print('fps: {}'.format(fps_window / (t1 - t0)))
                t0 = t1
                step = -1

    def create_sim(self):
        self.scene_setter(self.gym, self.sim)
        if self.asset_file_setter is None:

            def asset_file_setter_const(*args, **kwargs):
                return self.asset_file

            asset_file_setter = asset_file_setter_const
        else:
            asset_file_setter = self.asset_file_setter
        self.envs, self.actor_handles, self.assets = create_actors(
            asset_file_setter=asset_file_setter,
            asset_root=self.asset_root,
            asset_options_setter=self.asset_options_setter,
            asset_setter=self.asset_setter,
            pose_setter=self.pose_setter,
            gym=self.gym,
            sim=self.sim,
            collision_group_setter=self.collision_group_setter,
            collision_filter_setter=self.collision_filter_setter,
            spacing=self.spacing,
            num_envs=self.num_envs,
            num_env_actors=self.num_env_actors,
            actor_setter=self.actor_setter,
            env_setter=self.env_setter,
        )
